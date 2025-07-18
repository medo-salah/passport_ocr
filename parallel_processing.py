import concurrent.futures
from typing import List, Callable, Any, Dict, Optional
import logging
import os
from functools import partial

logger = logging.getLogger(__name__)

def parallel_process(
    items: List[Any],
    process_func: Callable,
    max_workers: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process items in parallel using a thread pool
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        max_workers: Maximum number of worker threads (default: CPU count)
        **kwargs: Additional arguments to pass to process_func
        
    Returns:
        List of processing results
    """
    # Determine number of workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 workers max
    
    logger.info(f"Starting parallel processing with {max_workers} workers for {len(items)} items")
    results = []
    
    # Create a partial function with the kwargs
    func = partial(process_func, **kwargs) if kwargs else process_func
    
    # Process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
                results.append({"status": "error", "error": str(e), "item": item})
    
    return results

async def async_parallel_process(
    items: List[Any],
    process_func: Callable,
    max_workers: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Async wrapper for parallel_process to use in FastAPI
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        max_workers: Maximum number of worker threads (default: CPU count)
        **kwargs: Additional arguments to pass to process_func
        
    Returns:
        List of processing results
    """
    # Run CPU-bound task in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: parallel_process(items, process_func, max_workers, **kwargs)
    )