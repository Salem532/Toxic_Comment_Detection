from src.evaluator import ToxicEvaluator
from src.utils.model_utils import load_config, seed_everything
import torch

def evaluate_test_set():
    print("=== 开始评估测试集 ===")
    config = load_config("configs/config.yaml")
    seed_everything(42)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"-> 使用设备: {device}")
    evaluator = ToxicEvaluator(config=config, device=device)
    evaluator.run()

if __name__ == "__main__":
    evaluate_test_set()




