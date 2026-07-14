import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from .config import DEVICE, PREDICT_CONFIG, MODELS_DIR, LLM_CONFIG 
from .lora_finetuner import ETFAdvisorLoRATuner

class TimeSeriesTransformer(nn.Module):
    """时间序列预测Transformer模型"""
    
    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        seq_length: int = 60,
        pred_length: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # 特征投影
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_length * input_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: [batch, seq, features]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后时间步
        x = self.output_proj(x)
        x = x.view(-1, self.pred_length, self.input_size)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ETFPricePredictor:
    """ETF价格预测器"""
    
    def __init__(self, base_model_name: Optional[str] = None):
        # 
        if base_model_name is None:
            base_model_name = LLM_CONFIG.get("model_name", "Qwen/Qwen3-30B-A3B")
        self.base_model_name = base_model_name

        self.model = TimeSeriesTransformer(
            seq_length=PREDICT_CONFIG["seq_length"],
            pred_length=PREDICT_CONFIG["pred_length"]
        ).to(DEVICE)
        
        self.model_path = PREDICT_CONFIG["model_path"]
        self.seq_length = PREDICT_CONFIG["seq_length"]
        self.pred_length = PREDICT_CONFIG["pred_length"]
        self.lora_path = PREDICT_CONFIG.get("lora_path", MODELS_DIR / "lora_etf_advisor")
        self.norm_params = {}
        self.is_trained = False
        
        # 尝试加载预训练模型
        if self.model_path.exists():
            self.load_model()
    
        if self.lora_path.exists():
            self.load_lora_adapter(str(self.lora_path))
            
    def load_model(self):
        """加载模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.norm_params = checkpoint.get('norm_params', {})
            self.is_trained = True
            print("预测模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
    
    def load_lora_adapter(self, lora_path: str):
        """加载微调后的 LoRA 适配器"""
        try:
            from peft import PeftModel
            
            if self.model is not None:
                # 
                if not hasattr(self.model, 'base_model'):
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    print(f"LoRA 适配器已加载: {lora_path}")
                else:
                    print(f"ℹ模型已加载 LoRA，跳过重复加载")
        except ImportError as e:
            print(f"⚠️ peft 未安装，跳过 LoRA 加载: {e}")
        except Exception as e:
            print(f"⚠️ LoRA 加载失败: {e}")

    def fine_tune(self, train_data: pd.DataFrame, output_dir: str = "./lora_etf_advisor"):
        """使用 LoRA 微调预测模型"""
        try:
            # 使用 self.base_model_name（从 LLM_CONFIG 获取）
            tuner = ETFAdvisorLoRATuner(self.base_model_name)
            tuner.load_model_and_tokenizer()
            tuner.train_lora(train_data, output_dir)
            
            # 微调完成后自动加载
            self.lora_path = Path(output_dir)
            if self.lora_path.exists():
                self.load_lora_adapter(str(self.lora_path))
                
        except ImportError as e:
            print(f"LoRA 微调失败，请安装 peft: {e}")
        except Exception as e:
            print(f"微调过程出错: {e}")
            
        
    def save_model(self):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'norm_params': self.norm_params,
        }
        torch.save(checkpoint, self.model_path)
        print(f"模型已保存: {self.model_path}")
    
    def prepare_data(self, df: pd.DataFrame) -> torch.Tensor:
        """准备输入数据"""
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        
        # 标准化
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1
        features_norm = (features - means) / stds
        
        self.norm_params['means'] = means
        self.norm_params['stds'] = stds
        
        return torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """预测未来价格"""
        if len(df) < self.seq_length:
            return {
                'error': f'数据不足，需要至少{self.seq_length}个交易日',
                'success': False
            }
        
        # 准备数据
        data_tensor = self.prepare_data(df)
        data_tensor = data_tensor.to(DEVICE)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data_tensor)
        
        # 反标准化
        pred_np = pred.cpu().numpy()[0]
        means = self.norm_params['means']
        stds = self.norm_params['stds']
        pred_denorm = pred_np * stds + means
        
        # 生成日期
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(self.pred_length)]
        
        # 计算置信区间
        recent_vol = df['Close'].pct_change().std() * np.sqrt(252)
        confidence = 1.96 * recent_vol * np.sqrt(self.pred_length / 252)
        
        close_prices = pred_denorm[:, 3]
        
        return {
            'success': True,
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'open': pred_denorm[:, 0].tolist(),
            'high': pred_denorm[:, 1].tolist(),
            'low': pred_denorm[:, 2].tolist(),
            'close': close_prices.tolist(),
            'confidence': confidence,
            'predicted_change': (close_prices[-1] - close_prices[0]) / close_prices[0],
        }
    
    def train(self, df_list: List[pd.DataFrame], epochs: int = 50) -> Dict:
        """训练模型"""
        sequences = []
        targets = []
        
        for df in df_list:
            if len(df) < self.seq_length + self.pred_length:
                continue
            
            data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            stds[stds == 0] = 1
            data_norm = (data - means) / stds
            
            for i in range(len(data_norm) - self.seq_length - self.pred_length + 1):
                sequences.append(data_norm[i:i+self.seq_length])
                targets.append(data_norm[i+self.seq_length:i+self.seq_length+self.pred_length])
        
        if not sequences:
            return {'error': '数据不足', 'success': False}
        
        X = torch.tensor(np.array(sequences), dtype=torch.float32).to(DEVICE)
        y = torch.tensor(np.array(targets), dtype=torch.float32).to(DEVICE)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.HuberLoss()
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
        
        self.is_trained = True
        self.save_model()
        
        return {
            'success': True,
            'losses': losses,
            'n_samples': len(sequences),
            'final_loss': losses[-1]
        }