# app/milvus_client.py
"""
Milvus 统一客户端 - 支持 Milvus Lite 自动降级
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

# ========== 尝试导入 Milvus Lite ==========
try:
    from milvus_lite import MilvusLite
    MILVUS_AVAILABLE = True
except ImportError:
    MilvusLite = None
    MILVUS_AVAILABLE = False
    logging.getLogger(__name__).warning("⚠️ milvus-lite 未安装，将使用内存模式")

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient as PyMilvusClient, DataType

from .config import MILVUS_CONFIG, RAG_CONFIG, BASE_DIR

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus 统一客户端 - 使用 PyMilvus 新 API
    优先使用 Milvus Lite，失败时自动降级到内存模式
    """
    
    _instance = None
    _client = None
    _memory_mode = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.collection_name = MILVUS_CONFIG.get("collection_name", "etf_knowledge")
        self.dim = MILVUS_CONFIG.get("dim", 384)
        self.top_k = MILVUS_CONFIG.get("top_k", 5)
        knowledge_dir_str = RAG_CONFIG.get("knowledge_dir", str(BASE_DIR / "knowledge"))
        self.knowledge_dir = Path(knowledge_dir_str)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Milvus Lite 数据目录
        self.milvus_data_dir = Path(BASE_DIR / "milvus_data.db")
        self.milvus_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 Embedding 模型
        try:
            self.embedder = SentenceTransformer(
                RAG_CONFIG.get("embedding_model", "all-MiniLM-L6-v2"),
                device="cpu"
            )
            logger.info("✅ Embedding 模型已加载")
        except Exception as e:
            logger.warning(f"⚠️ Embedding 模型加载失败: {e}")
            self.embedder = None
        
        # 初始化知识库
        self.knowledge = []
        self._load_knowledge()
        
        # 尝试启动 Milvus Lite
        if MILVUS_CONFIG.get("enabled", True) and MILVUS_AVAILABLE:
            self._init_milvus_lite()
        else:
            if not MILVUS_AVAILABLE:
                logger.info("📌 Milvus Lite 未安装，使用内存模式")
            else:
                logger.info("📌 Milvus 已禁用，使用内存模式")
            self._memory_mode = True
        
        # 如果 Milvus Lite 启动失败，使用内存模式
        if self._memory_mode:
            self._build_memory_index()
        
        logger.info(f"✅ Milvus 客户端初始化完成")
        logger.info(f"   📊 模式: {'Milvus Lite' if not self._memory_mode else '内存模式'}")
        logger.info(f"   📚 知识条目: {len(self.knowledge)}")
        logger.info(f"   📏 向量维度: {self.dim}")
    
    def _init_milvus_lite(self):
        """初始化 Milvus Lite - 使用新版 PyMilvus API"""
        try:
            uri = f"file:{self.milvus_db_path}"
            self._client = PyMilvusClient(uri=uri)
            logger.info(f"✅ Milvus Lite 已连接: {self.milvus_data_dir}")
            
            # 初始化 Collection
            self._init_collection()
            self._memory_mode = False
            
        except ImportError as e:
            logger.warning(f"⚠️ milvus-lite 未安装: {e}")
            logger.info("   💡 安装: pip install milvus-lite")
            self._memory_mode = True
        except Exception as e:
            logger.warning(f"⚠️ Milvus 初始化失败: {e}")
            self._memory_mode = True
    
    def _init_collection(self):
        """初始化 Milvus Collection"""
        # 检查 Collection 是否存在
        if self._client.has_collection(self.collection_name):
            logger.info(f"📂 使用已有 Collection: {self.collection_name}")
        else:
            # 创建 Schema
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
            schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            
            # 创建索引
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type=MILVUS_CONFIG.get("metric_type", "IP"),
                index_type=MILVUS_CONFIG.get("index_type", "IVF_FLAT"),
                params={"nlist": MILVUS_CONFIG.get("nlist", 128)}
            )
            
            # 创建 Collection
            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"✅ 创建 Collection: {self.collection_name}")
        
        # 如果 Collection 为空，插入数据
        if self.knowledge:
            stats = self._client.get_collection_stats(self.collection_name)
            if stats.get("row_count", 0) == 0:
                self._insert_knowledge(self.knowledge)
    
    def _load_knowledge(self):
        """加载知识库"""
        default_knowledge = [
            {"id": "etf_basics", "title": "ETF 基础知识", 
             "content": "ETF（交易型开放式指数基金）是在交易所上市交易的基金，可像股票一样买卖。", 
             "category": "基础"},
            {"id": "grid_trading", "title": "网格交易策略",
             "content": "网格交易是在设定价格区间内，通过分批买入和卖出获取收益的策略。", 
             "category": "策略"},
            {"id": "dca_strategy", "title": "定期定投策略",
             "content": "定期定额投资（DCA）是通过固定时间投入固定金额来平均成本的长线策略。", 
             "category": "策略"},
            {"id": "csi300", "title": "沪深300指数",
             "content": "沪深300指数由沪深两市规模最大、流动性最好的300只股票组成。", 
             "category": "指数"},
            {"id": "technical_analysis", "title": "技术分析基础",
             "content": "技术分析通过研究历史价格和成交量预测未来走势，常用指标有RSI、MACD、均线等。", 
             "category": "技术分析"},
            {"id": "risk_management", "title": "风险管理",
             "content": "风险管理包括仓位控制、止损设置、分散投资等方法，是投资成功的关键。", 
             "category": "风险管理"},
            {"id": "etf_types", "title": "ETF 类型",
             "content": "ETF 包括宽基ETF（如沪深300）、行业ETF（如证券ETF）、主题ETF（如新能源ETF）等。", 
             "category": "基础"},
            {"id": "macd_analysis", "title": "MACD 指标",
             "content": "MACD 由DIF、DEA和MACD柱组成，金叉（DIF上穿DEA）为买入信号，死叉为卖出信号。", 
             "category": "技术分析"},
            {"id": "rsi_analysis", "title": "RSI 指标",
             "content": "RSI 衡量价格变动速度，RSI>70为超买（可能回调），RSI<30为超卖（可能反弹）。", 
             "category": "技术分析"},
            {"id": "position_sizing", "title": "仓位管理",
             "content": "仓位管理根据风险承受能力和市场状况分配资金，建议单只ETF不超过总仓位的20%。", 
             "category": "风险管理"},
            {"id": "stop_loss", "title": "止损策略",
             "content": "止损是控制亏损的重要手段，常见止损方法包括固定比例止损（如-5%）、移动止损等。", 
             "category": "风险管理"},
        ]
        
        custom_knowledge = self._load_custom_knowledge()
        self.knowledge = default_knowledge + custom_knowledge
    
    def _load_custom_knowledge(self) -> List[Dict]:
        """加载自定义知识"""
        custom_knowledge = []
        for file_path in self.knowledge_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        custom_knowledge.extend(data)
                    elif isinstance(data, dict):
                        custom_knowledge.append(data)
                logger.info(f"  加载自定义知识: {file_path.name}")
            except Exception as e:
                logger.warning(f"  加载失败 {file_path.name}: {e}")
        return custom_knowledge
    
    def _insert_knowledge(self, knowledge: List[Dict]):
        """插入知识到 Milvus"""
        if self._memory_mode or self.embedder is None or not knowledge:
            return
        
        try:
            texts = [item['content'] for item in knowledge]
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            
            data = []
            for i, item in enumerate(knowledge):
                data.append({
                    "id": item.get('id', hashlib.md5(item['content'].encode()).hexdigest()[:16]),
                    "title": item.get('title', ''),
                    "content": item['content'],
                    "category": item.get('category', 'general'),
                    "embedding": embeddings[i].tolist(),
                    "created_at": datetime.now().isoformat(),
                    "metadata": json.dumps(item.get('metadata', {}))
                })
            
            self._client.insert(self.collection_name, data)
            logger.info(f"✅ 插入 {len(data)} 条知识")
        except Exception as e:
            logger.warning(f"⚠️ 插入知识失败: {e}")
            self._memory_mode = True
    
    def _build_memory_index(self):
        """构建内存索引（降级方案）"""
        if self.embedder is None or not self.knowledge:
            return
        
        try:
            texts = [item['content'] for item in self.knowledge]
            self._memory_embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            self._memory_embeddings = self._memory_embeddings / np.linalg.norm(
                self._memory_embeddings, axis=1, keepdims=True
            )
            logger.info("✅ 内存索引已构建")
        except Exception as e:
            logger.warning(f"⚠️ 内存索引构建失败: {e}")
            self._memory_embeddings = None
    
    # ============================================================
    # 公共接口
    # ============================================================
    
    def search(self, query: str, top_k: Optional[int] = None, category: Optional[str] = None) -> List[Dict]:
        """搜索知识"""
        top_k = top_k or self.top_k
        
        if self._memory_mode or self.embedder is None:
            return self._search_memory(query, top_k)
        
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            
            filter_expr = None
            if category:
                filter_expr = f'category == "{category}"'
            
            results = self._client.search(
                collection_name=self.collection_name,
                data=query_embedding.tolist(),
                limit=top_k,
                filter=filter_expr,
                output_fields=["id", "title", "content", "category", "metadata"],
                search_params={
                    "metric_type": MILVUS_CONFIG.get("metric_type", "IP"),
                    "params": {"nprobe": 16}
                }
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    if hit['distance'] > 0.3:
                        entity = hit['entity']
                        formatted_results.append({
                            "id": entity.get('id'),
                            "title": entity.get('title', ''),
                            "content": entity.get('content', ''),
                            "category": entity.get('category', 'general'),
                            "score": hit['distance'],
                            "metadata": json.loads(entity.get('metadata', '{}')) if entity.get('metadata') else {}
                        })
            
            if len(formatted_results) < 2:
                keyword_results = self._keyword_search(query)
                existing_ids = {r['id'] for r in formatted_results}
                for item in keyword_results:
                    if item['id'] not in existing_ids:
                        formatted_results.append(item)
            
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Milvus 搜索失败，降级到内存: {e}")
            return self._search_memory(query, top_k)
    
    def _search_memory(self, query: str, top_k: int) -> List[Dict]:
        """内存模式搜索"""
        if not self.knowledge or self.embedder is None:
            return []
        
        try:
            if not hasattr(self, '_memory_embeddings') or self._memory_embeddings is None:
                self._build_memory_index()
            
            if self._memory_embeddings is None:
                return self._keyword_search(query)
            
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            similarities = np.dot(self._memory_embeddings, query_embedding.T).flatten()
            indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in indices:
                if similarities[idx] > 0.3:
                    item = self.knowledge[idx].copy()
                    item['score'] = float(similarities[idx])
                    results.append(item)
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ 内存搜索失败: {e}")
            return self._keyword_search(query)
    
    def _keyword_search(self, query: str) -> List[Dict]:
        """关键词搜索（降级）"""
        query_lower = query.lower()
        keywords = query_lower.split()
        
        results = []
        for item in self.knowledge:
            content_lower = item.get('content', '').lower()
            title_lower = item.get('title', '').lower()
            
            score = 0
            for kw in keywords:
                if kw in content_lower:
                    score += 2
                if kw in title_lower:
                    score += 1
            
            if score > 0:
                item = item.copy()
                item['score'] = min(score / 10, 0.9)
                results.append(item)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:self.top_k]
    
    def insert(self, title: str, content: str, category: str = "general", metadata: Optional[Dict] = None) -> str:
        """插入知识"""
        item_id = hashlib.md5(f"{title}{content}{datetime.now()}".encode()).hexdigest()[:16]
        
        new_item = {
            "id": item_id,
            "title": title,
            "content": content,
            "category": category,
            "metadata": metadata or {}
        }
        
        self.knowledge.append(new_item)
        
        # 如果使用 Milvus Lite，插入到 Milvus
        if not self._memory_mode and self.embedder is not None:
            try:
                embedding = self.embedder.encode([content], convert_to_numpy=True)
                self._client.insert(self.collection_name, [{
                    "id": item_id,
                    "title": title,
                    "content": content,
                    "category": category,
                    "embedding": embedding[0].tolist(),
                    "created_at": datetime.now().isoformat(),
                    "metadata": json.dumps(metadata or {}),
                }])
            except Exception as e:
                logger.warning(f"⚠️ 插入失败: {e}")
                self._memory_mode = True
                self._build_memory_index()
        else:
            # 内存模式，重建索引
            self._build_memory_index()
        
        # 保存到文件
        self._save_custom_knowledge(new_item)
        return item_id
    
    def _save_custom_knowledge(self, item: Dict):
        """保存自定义知识"""
        file_path = self.knowledge_dir / "custom_knowledge.json"
        try:
            existing = []
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            existing.append(item)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存知识失败: {e}")
    
    def delete(self, item_id: str) -> bool:
        """删除知识"""
        if not self._memory_mode:
            try:
                self._client.delete(self.collection_name, f'id == "{item_id}"')
            except Exception as e:
                logger.warning(f"⚠️ 删除失败: {e}")
        
        self.knowledge = [item for item in self.knowledge if item['id'] != item_id]
        
        if self._memory_mode:
            self._build_memory_index()
        
        return True
    
    def delete_all(self):
        """清空所有知识"""
        if not self._memory_mode:
            try:
                self._client.delete(self.collection_name, "id is not None")
            except Exception as e:
                logger.warning(f"⚠️ 清空失败: {e}")
        
        self.knowledge = []
        if self._memory_mode:
            self._memory_embeddings = None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "collection": self.collection_name,
            "dimension": self.dim,
            "total_knowledge": len(self.knowledge),
            "top_k": self.top_k,
            "memory_mode": self._memory_mode,
            "data_dir": str(self.milvus_data_dir) if hasattr(self, 'milvus_data_dir') else None,
        }
        
        if not self._memory_mode and self._client:
            try:
                collection_stats = self._client.get_collection_stats(self.collection_name)
                stats["total_entities"] = collection_stats.get("row_count", 0)
            except Exception as e:
                logger.debug(f"获取统计信息失败: {e}")
        
        return stats
    
    def stop(self):
        """停止 Milvus Lite 服务"""
        if self._client:
            try:
                self._client.close()
                logger.info("✅ Milvus Lite 已停止")
            except Exception as e:
                logger.warning(f"停止 Milvus 失败: {e}")


# 全局单例
def get_milvus_client():
    """获取 Milvus 客户端"""
    return MilvusClient()


# 兼容旧代码
MilvusRAG = MilvusClient