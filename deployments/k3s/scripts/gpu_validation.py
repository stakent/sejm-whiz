#!/usr/bin/env python3
"""
GPU Validation Script for sejm-whiz processor
Based on k3s-docker-gpu example, adapted for HerBERT embeddings testing
"""

import os
import sys
import subprocess
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment() -> None:
    """Test basic environment and Python setup"""
    logger.info("=== Testing Environment ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check NVIDIA environment variables
    nvidia_vars = ["NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES"]
    for var in nvidia_vars:
        value = os.getenv(var, "Not set")
        logger.info(f"{var}: {value}")

def test_nvidia_smi() -> bool:
    """Test nvidia-smi availability and output"""
    logger.info("=== Testing nvidia-smi ===")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        logger.info("nvidia-smi output:")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("nvidia-smi not found")
        return False

def test_torch_cuda() -> bool:
    """Test PyTorch CUDA availability"""
    logger.info("=== Testing PyTorch CUDA ===")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA device count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"Device {i}: {device_name}")
                
                # Test memory
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                logger.info(f"Device {i} memory - Allocated: {memory_allocated}, Reserved: {memory_reserved}")
                
            # Simple CUDA operation test
            logger.info("Testing CUDA tensor operations...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.mm(x, y)
            logger.info(f"CUDA tensor operation successful. Result shape: {z.shape}")
            
        return cuda_available
        
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        return False
    except Exception as e:
        logger.error(f"PyTorch CUDA test failed: {e}")
        return False

def test_herbert_embeddings() -> bool:
    """Test HerBERT embeddings with GPU"""
    logger.info("=== Testing HerBERT Embeddings ===")
    try:
        # Test embeddings component import
        sys.path.append('/app')
        from components.sejm_whiz.embeddings.herbert_encoder import HerBERTEncoder
        from components.sejm_whiz.embeddings.config import EmbeddingsConfig
        
        logger.info("HerBERT components imported successfully")
        
        # Test configuration
        config = EmbeddingsConfig()
        logger.info(f"Embedding device configured: {config.device}")
        
        # Test encoder initialization (this will test actual GPU access)
        encoder = HerBERTEncoder(config)
        logger.info("HerBERT encoder initialized successfully")
        
        # Test encoding a simple text
        test_text = "Test dokument prawny dla sprawdzenia embeddings."
        embeddings = encoder.encode([test_text])
        
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Sample embedding values: {embeddings[0][:5]}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import HerBERT components: {e}")
        return False
    except Exception as e:
        logger.error(f"HerBERT embedding test failed: {e}")
        return False

def main() -> None:
    """Main GPU validation routine"""
    logger.info("Starting GPU validation for sejm-whiz processor")
    
    # Run all tests
    tests = [
        ("Environment", test_environment, True),  # Always run
        ("NVIDIA SMI", test_nvidia_smi, False),
        ("PyTorch CUDA", test_torch_cuda, False),
        ("HerBERT Embeddings", test_herbert_embeddings, False),
    ]
    
    results = {}
    
    for test_name, test_func, always_run in tests:
        try:
            if always_run:
                test_func()
                results[test_name] = True
            else:
                result = test_func()
                results[test_name] = result
                if not result:
                    logger.warning(f"{test_name} test failed")
        except Exception as e:
            logger.error(f"{test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=== GPU Validation Summary ===")
    passed = 0
    total = len([t for t in tests if not t[2]])  # Don't count always_run tests
    
    for test_name, _, always_run in tests:
        if not always_run:
            status = "PASS" if results.get(test_name, False) else "FAIL"
            logger.info(f"{test_name}: {status}")
            if results.get(test_name, False):
                passed += 1
    
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All GPU tests passed! GPU acceleration ready.")
        sys.exit(0)
    else:
        logger.error("❌ Some GPU tests failed. Check configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()