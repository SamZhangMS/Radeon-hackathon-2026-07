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

class LSTMPredictor(nn.Module):
    """LSTM 时间序列预测模型 - 作为第二种预测模型"""
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        seq_length: int = 60,
        pred_length: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_length * input_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        output = output.view(-1, self.pred_length, self.input_size)
        return output

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
        
        self.lstm_model = LSTMPredictor(
            seq_length=PREDICT_CONFIG["seq_length"],
            pred_length=PREDICT_CONFIG["pred_length"]
        ).to(DEVICE)

        self.model_path = PREDICT_CONFIG["model_path"]
        self.lstm_model_path = MODELS_DIR / "etf_predictor_lstm.pt"
        self.seq_length = PREDICT_CONFIG["seq_length"]
        self.pred_length = PREDICT_CONFIG["pred_length"]
        self.lora_path = PREDICT_CONFIG.get("lora_path", MODELS_DIR / "lora_etf_advisor")
        self.norm_params = {}
        self.is_trained = False
        
        # 尝试加载预训练模型
        if self.model_path.exists():
            self.load_model()
    
        if self.lstm_model_path.exists():
            self.load_lstm_model()

        if self.lora_path.exists():
            self.load_lora_adapter(str(self.lora_path))
            
    def load_lstm_model(self):
        """加载 LSTM 模型"""
        try:
            checkpoint = torch.load(self.lstm_model_path, map_location=DEVICE)
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            print("LSTM 预测模型加载成功")
        except Exception as e:
            print(f"⚠️ LSTM 模型加载失败: {e}")

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
    
    def save_lstm_model(self):
        """保存 LSTM 模型"""
        checkpoint = {
            'model_state_dict': self.lstm_model.state_dict(),
            'norm_params': self.norm_params,
        }
        torch.save(checkpoint, self.lstm_model_path)
        print(f"LSTM 模型已保存: {self.lstm_model_path}")

    def predict_ensemble(self, df: pd.DataFrame) -> Dict:
        """✅ 双模型集成预测（新增）
        
        使用 Transformer + LSTM 两种模型进行预测，取加权平均
        """
        if len(df) < self.seq_length:
            return {
                'error': f'数据不足，需要至少{self.seq_length}个交易日',
                'success': False
            }
        
        # 准备数据
        data_tensor = self.prepare_data(df)
        data_tensor = data_tensor.to(DEVICE)
        
        # 1. Transformer 预测
        self.transformer_model.eval()
        with torch.no_grad():
            pred_trans = self.transformer_model(data_tensor)
        
        # 2. LSTM 预测
        self.lstm_model.eval()
        with torch.no_grad():
            pred_lstm = self.lstm_model(data_tensor)
        
        # 3. 反标准化
        pred_trans_np = pred_trans.cpu().numpy()[0]
        pred_lstm_np = pred_lstm.cpu().numpy()[0]
        means = self.norm_params['means']
        stds = self.norm_params['stds']
        
        pred_trans_denorm = pred_trans_np * stds + means
        pred_lstm_denorm = pred_lstm_np * stds + means
        
        # 4. 集成预测（加权平均，Transformer 权重 0.6，LSTM 权重 0.4）
        ensemble_weight = 0.6
        pred_ensemble = pred_trans_denorm * ensemble_weight + pred_lstm_denorm * (1 - ensemble_weight)
        
        # 生成日期
        last_date = df.index[-1]
        if isinstance(last_date, (int, float)):
            try:
                last_date = pd.to_datetime(last_date, unit='s')
            except:
                last_date = datetime.now()
        elif not isinstance(last_date, pd.Timestamp):
            last_date = datetime.now()
    

        future_dates = [last_date + timedelta(days=i+1) for i in range(self.pred_length)]
        
        # 计算置信区间
        recent_vol = df['close'].pct_change().std() * np.sqrt(252)
        confidence = 1.96 * recent_vol * np.sqrt(self.pred_length / 252)
        
        close_prices = pred_ensemble[:, 3]
        
        return {
            'success': True,
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'open': pred_ensemble[:, 0].tolist(),
            'high': pred_ensemble[:, 1].tolist(),
            'low': pred_ensemble[:, 2].tolist(),
            'close': close_prices.tolist(),
            'confidence': confidence,
            'predicted_change': (close_prices[-1] - close_prices[0]) / close_prices[0],
            # ✅ 新增：各模型独立预测结果
            'transformer_prediction': {
                'close': pred_trans_denorm[:, 3].tolist(),
                'change': (pred_trans_denorm[-1, 3] - pred_trans_denorm[0, 3]) / pred_trans_denorm[0, 3]
            },
            'lstm_prediction': {
                'close': pred_lstm_denorm[:, 3].tolist(),
                'change': (pred_lstm_denorm[-1, 3] - pred_lstm_denorm[0, 3]) / pred_lstm_denorm[0, 3]
            },
            'ensemble_weight': ensemble_weight,
        }

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
        features = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        
        # 标准化
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1
        features_norm = (features - means) / stds
        
        self.norm_params['means'] = means
        self.norm_params['stds'] = stds
        
        return torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
    
    def predict(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """预测未来价格（默认使用双模型集成）"""
        if use_ensemble and self.lstm_model_path.exists():
            return self.predict_ensemble(df)
        else:
            return self.predict_single(df)
    
    def predict_single(self, df: pd.DataFrame) -> Dict:
        """预测未来价格"""
        if len(df) < self.seq_length:
            return {
                'error': f'数据不足，需要至少{self.seq_length}个交易日',
                'success': False
            }
        
        # 准备数据
        data_tensor = self.prepare_data(df)
        model_dtype = next(self.model.parameters()).dtype
        data_tensor = data_tensor.to(DEVICE).to(model_dtype)
        
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
        if isinstance(last_date, (int, float)):
            try:
                last_date = pd.to_datetime(last_date, unit='s')
            except:
                last_date = datetime.now()
        elif not isinstance(last_date, pd.Timestamp):
            last_date = datetime.now()
        
        future_dates = [last_date + timedelta(days=i+1) for i in range(self.pred_length)]
        
        # 计算置信区间
        recent_vol = df['close'].pct_change().std() * np.sqrt(252)
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
    
    def train_lstm(self, df_list: List[pd.DataFrame], epochs: int = 50) -> Dict:
        """训练 LSTM 模型"""
        sequences = []
        targets = []
        
        for df in df_list:
            if len(df) < self.seq_length + self.pred_length:
                continue
            
            data = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.HuberLoss()
        
        self.lstm_model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.lstm_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
        
        self.save_lstm_model()
        
        return {
            'success': True,
            'losses': losses,
            'n_samples': len(sequences),
            'final_loss': losses[-1]
        }
        
    def train(self, df_list: List[pd.DataFrame], epochs: int = 50) -> Dict:
        """训练模型"""
        sequences = []
        targets = []
        
        for df in df_list:
            if len(df) < self.seq_length + self.pred_length:
                continue
            
            data = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
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