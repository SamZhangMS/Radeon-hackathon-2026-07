# app/milvus_client.py
"""
Milvus 统一客户端 - 支持三种数据类型的存储
使用基类 + 子类的设计模式，代码更简洁
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# ========== 尝试导入 Milvus ==========
try:
    from pymilvus import MilvusClient as PyMilvusClient, DataType
    PYMILVUS_AVAILABLE = True
except ImportError:
    PyMilvusClient = None
    DataType = None
    PYMILVUS_AVAILABLE = False
    logging.getLogger(__name__).warning("⚠️ pymilvus 未安装")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .config import MILVUS_CONFIG, RAG_CONFIG, BASE_DIR

logger = logging.getLogger(__name__)


# ============================================================
# 基类：Milvus 连接管理
# ============================================================

class MilvusConnection:
    """
    Milvus 连接管理基类
    负责连接、初始化、降级等公共功能
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
        
        # Milvus 数据目录
        self.milvus_data_dir = BASE_DIR / "milvus_data.db"
        self.milvus_data_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.milvus_data_dir.exists():
            self.milvus_data_dir.touch()
            
        self.milvus_data_path = str(self.milvus_data_dir.absolute())
        
        logger.info(f"📁 Milvus 数据目录: {self.milvus_data_path}")
        
        # 尝试启动 Milvus Lite
        if MILVUS_CONFIG.get("enabled", True) and PYMILVUS_AVAILABLE:
            self._init_milvus_lite()
        else:
            self._memory_mode = True
            if not PYMILVUS_AVAILABLE:
                logger.info("📌 pymilvus 未安装，使用内存模式")
            else:
                logger.info("📌 Milvus 已禁用，使用内存模式")
    
    def _init_milvus_lite(self):
        """初始化 Milvus Lite"""
        if PyMilvusClient is None:
            self._memory_mode = True
            return
        
        try:
            # ✅ 修复：使用正确的 URI 格式
            # Milvus Lite 要求 file: 前缀加绝对路径
            uri = f"{self.milvus_data_path}"
            
            logger.info(f"🔗 连接 Milvus Lite: {uri}")
            
            self._client = PyMilvusClient(
                uri=uri,
                timeout=60,
                keepalive_options={
                    'keepalive_time_ms': 60000,
                    'keepalive_timeout_ms': 20000,
                    'keepalive_permit_without_calls': True,
                }
            )
            
            # ✅ 测试连接
            try:
                # 尝试获取版本信息或执行简单操作来验证连接
                self._client.get_collection_stats("dummy")
                logger.info(f"✅ Milvus Lite 已连接: {uri}")
                self._memory_mode = False
            except Exception as e:
                logger.warning(f"⚠️ Milvus Lite 连接测试失败: {e}")
                self._memory_mode = True
                self._client = None
            
        except Exception as e:
            logger.warning(f"⚠️ Milvus 初始化失败: {e}")
            self._memory_mode = True
            self._client = None
    
    def _is_available(self) -> bool:
        """检查 Milvus 是否可用"""
        return (not self._memory_mode and 
                self._client is not None and
                PYMILVUS_AVAILABLE)
    
    def _ensure_collection(self, collection_name: str, schema_func, index_func=None):
        """确保 Collection 存在"""
        if not self._is_available():
            return False
        
        try:
            if self._client.has_collection(collection_name):
                return True
            
            # 创建 Schema
            schema = schema_func()
            
            # 创建索引
            index_params = None
            if index_func:
                index_params = index_func()
            
            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"✅ 创建 Collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 创建 Collection 失败: {e}")
            return False
    
    def _insert_data(self, collection_name: str, data: List[Dict]) -> bool:
        """插入数据"""
        if not self._is_available() or not data:
            return False
        
        try:
            self._client.insert(collection_name, data)
            return True
        except Exception as e:
            logger.warning(f"⚠️ 插入数据失败: {e}")
            return False
    
    def _query_data(self, collection_name: str, expr: str, 
                    output_fields: List[str], limit: int = 1000) -> List[Dict]:
        """查询数据"""
        if not self._is_available():
            return []
        
        try:
            return self._client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
                limit=limit
            )
        except Exception as e:
            logger.debug(f"查询失败: {e}")
            return []
    
    def _delete_data(self, collection_name: str, expr: str) -> bool:
        """删除数据"""
        if not self._is_available():
            return False
        
        try:
            self._client.delete(collection_name=collection_name, filter=expr)
            return True
        except Exception as e:
            logger.warning(f"⚠️ 删除失败: {e}")
            return False
    
    def _update_data(self, collection_name: str, expr: str, data: Dict) -> bool:
        """更新数据"""
        if not self._is_available():
            return False
        
        try:
            self._client.update(
                collection_name=collection_name,
                filter=expr,
                data=data
            )
            return True
        except Exception as e:
            logger.warning(f"⚠️ 更新失败: {e}")
            return False
    
    def _get_stats(self, collection_name: str) -> Dict:
        """获取 Collection 统计信息"""
        if not self._is_available():
            return {"row_count": 0, "error": "Milvus not available"}
        
        try:
            return self._client.get_collection_stats(collection_name)
        except Exception as e:
            return {"row_count": 0, "error": str(e)}
    
    def stop(self):
        """停止 Milvus 服务"""
        if self._client:
            try:
                self._client.close()
                logger.info("✅ Milvus 已停止")
            except Exception as e:
                logger.warning(f"停止 Milvus 失败: {e}")


# ============================================================
# 基类：Collection 管理器
# ============================================================

class BaseCollectionManager(ABC):
    """
    Collection 管理器基类
    每个数据类型继承此类，实现自己的 Schema 和 CRUD 方法
    """
    
    def __init__(self, collection_name: str):
        self.connection = MilvusConnection()
        self.collection_name = collection_name
        self._ensure_collection()
    
    @abstractmethod
    def _create_schema(self):
        """创建 Schema（子类必须实现）"""
        pass
    
    @abstractmethod
    def _create_index_params(self):
        """创建索引参数（子类可选实现）"""
        return None
    
    def _ensure_collection(self):
        """确保 Collection 存在"""
        return self.connection._ensure_collection(
            self.collection_name,
            self._create_schema,
            self._create_index_params
        )
    
    def insert(self, data: List[Dict]) -> bool:
        """插入数据"""
        return self.connection._insert_data(self.collection_name, data)
    
    def query(self, expr: str, output_fields: List[str], limit: int = 1000) -> List[Dict]:
        """查询数据"""
        return self.connection._query_data(self.collection_name, expr, output_fields, limit)
    
    def delete(self, expr: str) -> bool:
        """删除数据"""
        return self.connection._delete_data(self.collection_name, expr)
    
    def update(self, expr: str, data: Dict) -> bool:
        """更新数据"""
        return self.connection._update_data(self.collection_name, expr, data)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.connection._get_stats(self.collection_name)


# ============================================================
# 1. 知识库管理器（Knowledge）
# ============================================================

class KnowledgeManager(BaseCollectionManager):
    """知识库管理器 - 支持向量检索"""
    
    def __init__(self):
        self.dim = MILVUS_CONFIG.get("dim", 384)
        collection_name = MILVUS_CONFIG.get("collection_name", "etf_knowledge")
        super().__init__(collection_name)
        
        # 加载 Embedding 模型
        self.embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(
                    RAG_CONFIG.get("embedding_model", "all-MiniLM-L6-v2"),
                    device="cpu"
                )
            except Exception as e:
                logger.warning(f"⚠️ Embedding 模型加载失败: {e}")
        
        # 内存模式下的知识库缓存
        self._knowledge_cache = []
        self._memory_embeddings = None
        self._load_knowledge()
        
        # 如果是内存模式，构建索引
        if self.connection._memory_mode:
            self._build_memory_index()
    
    def _create_schema(self):
        """创建知识库 Schema"""
        if self.connection._client is None:
            return None
        
        schema = self.connection._client.create_schema(
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
        return schema
    
    def _create_index_params(self):
        """创建索引参数"""
        if self.connection._client is None:
            return None
        
        index_params = self.connection._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type=MILVUS_CONFIG.get("metric_type", "IP"),
            index_type=MILVUS_CONFIG.get("index_type", "IVF_FLAT"),
            params={"nlist": MILVUS_CONFIG.get("nlist", 128)}
        )
        return index_params
    
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
        
        # 加载自定义知识
        knowledge_dir = Path(RAG_CONFIG.get("knowledge_dir", str(BASE_DIR / "knowledge")))
        custom_knowledge = []
        for file_path in knowledge_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        custom_knowledge.extend(data)
                    elif isinstance(data, dict):
                        custom_knowledge.append(data)
            except Exception as e:
                logger.warning(f"加载自定义知识失败 {file_path.name}: {e}")
        
        self._knowledge_cache = default_knowledge + custom_knowledge
    
    def _build_memory_index(self):
        """构建内存索引（降级方案）"""
        if self.embedder is None or not self._knowledge_cache:
            return
        
        try:
            texts = [item['content'] for item in self._knowledge_cache]
            self._memory_embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            self._memory_embeddings = self._memory_embeddings / np.linalg.norm(
                self._memory_embeddings, axis=1, keepdims=True
            )
            logger.info("✅ 内存索引已构建")
        except Exception as e:
            logger.warning(f"⚠️ 内存索引构建失败: {e}")
            self._memory_embeddings = None
    
    def search(self, query: str, top_k: Optional[int] = None, 
               category: Optional[str] = None) -> List[Dict]:
        """搜索知识库"""
        top_k = top_k or MILVUS_CONFIG.get("top_k", 5)
        
        # 内存模式
        if self.connection._memory_mode or self.embedder is None:
            return self._search_memory(query, top_k)
        
        # Milvus 模式
        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            
            filter_expr = None
            if category:
                filter_expr = f'category == "{category}"'
            
            results = self.connection._client.search(
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
            
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️ Milvus 搜索失败，降级到内存: {e}")
            return self._search_memory(query, top_k)
    
    def _search_memory(self, query: str, top_k: int) -> List[Dict]:
        """内存模式搜索"""
        if not self._knowledge_cache or self.embedder is None:
            return []
        
        try:
            if self._memory_embeddings is None:
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
                    item = self._knowledge_cache[idx].copy()
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
        top_k = MILVUS_CONFIG.get("top_k", 5)
        
        results = []
        for item in self._knowledge_cache:
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
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def insert(self, title: str, content: str, category: str = "general", 
               metadata: Optional[Dict] = None) -> str:
        """插入知识"""
        item_id = hashlib.md5(f"{title}{content}{datetime.now()}".encode()).hexdigest()[:16]
        
        new_item = {
            "id": item_id,
            "title": title,
            "content": content,
            "category": category,
            "metadata": metadata or {}
        }
        
        self._knowledge_cache.append(new_item)
        
        if self.connection._is_available() and self.embedder is not None:
            try:
                embedding = self.embedder.encode([content], convert_to_numpy=True)
                self.connection._insert_data(self.collection_name, [{
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
                self._build_memory_index()
        else:
            self._build_memory_index()
        
        return item_id
    
    def delete(self, item_id: str) -> bool:
        """删除知识"""
        if self.connection._is_available():
            self.connection._delete_data(self.collection_name, f'id == "{item_id}"')
        
        self._knowledge_cache = [item for item in self._knowledge_cache if item['id'] != item_id]
        
        if self.connection._memory_mode:
            self._build_memory_index()
        
        return True

    def delete_by_id(self, item_id: str) -> bool:
        """删除知识"""
        if self.connection._is_available():
            self.connection._delete_data(self.collection_name, f'id == "{item_id}"')
        
        self._knowledge_cache = [item for item in self._knowledge_cache if item['id'] != item_id]
        
        if self.connection._memory_mode:
            self._build_memory_index()
        
        return True
    
    def delete_all(self) -> bool:
        """清空所有知识"""
        if self.connection._is_available():
            self.connection._delete_data(self.collection_name, "id is not None")
        
        self._knowledge_cache = []
        self._memory_embeddings = None
        
        return True
# ============================================================
# 2. 推荐缓存管理器（Recommendation）
# ============================================================

class RecommendationCacheManager(BaseCollectionManager):
    """推荐缓存管理器"""
    
    def __init__(self):
        collection_name = "etf_recommendation_cache"
        super().__init__(collection_name)
    
    def _create_schema(self):
        """创建推荐缓存 Schema"""
        if self.connection._client is None:
            return None

        schema = self.connection._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="symbol", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="analysis_type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="latest_date", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="result", datatype=DataType.JSON)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=30)
        schema.add_field(field_name="updated_at", datatype=DataType.VARCHAR, max_length=30)
        return schema
    
    def _create_index_params(self):
        """创建索引参数"""
        if self.connection._client is None:
            return None
        
        index_params = self.connection._client.prepare_index_params()
        index_params.add_index(field_name="symbol", index_type="INVERTED")
        index_params.add_index(field_name="analysis_type", index_type="INVERTED")
        return index_params
    
    def get(self, symbol: str, analysis_type: str) -> Optional[Dict]:
        """获取缓存"""
        expr = f'symbol == "{symbol}" and analysis_type == "{analysis_type}"'
        results = self.query(expr, ["latest_date", "result"], limit=1)
        
        if results:
            return {
                "latest_date": results[0]["latest_date"],
                "result": results[0]["result"]
            }
        return None
    
    def save(self, symbol: str, analysis_type: str, latest_date: str, result: Dict):
        """保存缓存"""
        now = datetime.now().isoformat()
        expr = f'symbol == "{symbol}" and analysis_type == "{analysis_type}"'
        
        existing = self.query(expr, ["id"], limit=1)
        
        if existing:
            self.update(expr, {
                "latest_date": latest_date,
                "result": result,
                "updated_at": now
            })
        else:
            self.insert([{
                "symbol": symbol,
                "analysis_type": analysis_type,
                "latest_date": latest_date,
                "result": result,
                "created_at": now,
                "updated_at": now
            }])
    
    def clear(self, symbol: Optional[str] = None, analysis_type: Optional[str] = None):
        """清除缓存"""
        if symbol and analysis_type:
            expr = f'symbol == "{symbol}" and analysis_type == "{analysis_type}"'
        elif symbol:
            expr = f'symbol == "{symbol}"'
        else:
            expr = "id is not None"
        
        self.delete(expr)


# ============================================================
# 3. 反馈管理器（Feedback）
# ============================================================

class FeedbackManager(BaseCollectionManager):
    """反馈管理器"""
    
    def __init__(self):
        collection_name = "etf_feedback"
        super().__init__(collection_name)
    
    def _create_schema(self):
        """创建反馈 Schema"""
        if self.connection._client is None:
            return None
        
        schema = self.connection._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=30)
        schema.add_field(field_name="symbol", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="recommendation", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="actual_result", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="user_rating", datatype=DataType.INT64)
        schema.add_field(field_name="user_comment", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="accuracy", datatype=DataType.FLOAT)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        return schema
    
    def _create_index_params(self):
        """创建索引参数"""
        if self.connection._client is None:
            return None
        index_params = self.connection._client.prepare_index_params()
        index_params.add_index(field_name="symbol", index_type="INVERTED")
        index_params.add_index(field_name="timestamp", index_type="INVERTED")
        index_params.add_index(field_name="user_rating", index_type="INVERTED")
        return index_params
    
    def insert_feedback(self, feedback: Dict) -> bool:
        """插入反馈"""
        # 确保必需字段存在
        required_fields = ["symbol", "recommendation", "actual_result", "user_rating"]
        for field in required_fields:
            if field not in feedback:
                logger.warning(f"⚠️ 反馈数据缺少必需字段: {field}")
                return False
        
        # 添加时间戳
        if "timestamp" not in feedback:
            feedback["timestamp"] = datetime.now().isoformat()
        
        # 计算准确率（如果未提供）
        if "accuracy" not in feedback:
            feedback["accuracy"] = 0.5
        
        return self.insert([feedback])
    
    def get_by_symbol(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """获取指定股票的反馈"""
        expr = f'symbol == "{symbol}"'
        return self.query(expr, [
            "timestamp", "recommendation", "actual_result", 
            "user_rating", "user_comment", "accuracy", "metadata"
        ], limit=limit)
    
    def get_all(self, limit: int = 10000) -> List[Dict]:
        """获取所有反馈"""
        return self.query("id is not None", [
            "symbol", "accuracy", "user_rating", "timestamp"
        ], limit=limit)
    
    def clear(self, symbol: Optional[str] = None):
        """清除反馈"""
        expr = f'symbol == "{symbol}"' if symbol else "id is not None"
        self.delete(expr)


# ============================================================
# 统一客户端接口
# ============================================================

class MilvusClient:
    """
    Milvus 统一客户端
    提供三个子管理器的访问入口
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # 初始化三个管理器
        self.knowledge = KnowledgeManager()
        self.recommendation = RecommendationCacheManager()
        self.feedback = FeedbackManager()
        
        logger.info("✅ Milvus 客户端初始化完成")
        logger.info(f"   📚 Collections: knowledge, recommendation, feedback")
    
    
    def get_stats(self) -> Dict:
        """获取所有 Collection 的统计信息"""
        return {
            "knowledge": self.knowledge.get_stats(),
            "recommendation": self.recommendation.get_stats(),
            "feedback": self.feedback.get_stats(),
            "mode": "Memory" if self.knowledge.connection._memory_mode else "Milvus Lite"
        }
    
    def stop(self):
        """停止 Milvus 服务"""
        self.knowledge.connection.stop()


# ============================================================
# 全局单例
# ============================================================

def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端"""
    return MilvusClient()