# Add these imports at the top
import gc
import psutil
import numpy as np
from threading import Lock
from functools import wraps
import torch
class MemoryManager:
    def __init__(self):
        self.lock = Lock()
        
    def check_gpu_memory(self):
        """Check GPU memory usage and return percentage used"""
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
            
        with self.lock:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            percent_used = memory_allocated / memory_total
            return percent_used, memory_allocated, memory_total
    
    def cleanup_memory(self, force=False):
        """Clean up GPU memory"""
        if not torch.cuda.is_available():
            return False, 0
            
        percent_used, allocated, total = self.check_gpu_memory()
        
        with self.lock:
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Second check after cleanup
            new_percent, new_allocated, _ = self.check_gpu_memory()
            freed = allocated - new_allocated
            
            return True, freed

# Create an instance of MemoryManager
memory_manager = MemoryManager()

# Create a memory management decorator
def manage_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clean up before processing
        memory_manager.cleanup_memory()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Clean up after processing
            memory_manager.cleanup_memory()
            
            return result
        except Exception as e:
            # Clean up on error
            memory_manager.cleanup_memory(force=True)
            raise e
    return wrapper