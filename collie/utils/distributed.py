from contextlib import contextmanager
import time

from accelerate import Accelerator


@contextmanager
def fix_file_not_found_in_fsdp(accelerator: Accelerator, offset_seconds: int = 1):
    time.sleep(int(accelerator.local_process_index) * offset_seconds)
    yield
    accelerator.wait_for_everyone()
