#!/usr/bin/env python3
"""
Prime Number Searcher

An efficient prime number search tool using the Sieve of Eratosthenes algorithm.
Saves prime numbers to primes.txt with an intuitive progress bar.
"""

import sys
import math
import time
import psutil
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()


class SystemMonitor:
    """Monitor CPU, RAM, and GPU usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.max_cpu = 0.0
        self.max_mem_mb = 0.0
        self.max_mem_pct = 0.0
        self.max_gpu_util = 0.0
        self.max_gpu_mem_mb = 0.0
        
        # Check for GPU availability
        self.has_gpu = False
        self.gpu_handles = []
        try:
            # Using nvidia-ml-py (not the deprecated pynvml package)
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
            self.has_gpu = len(self.gpu_handles) > 0
            self.pynvml = pynvml
        except (ImportError, Exception):
            self.has_gpu = False
        
        # Initialize CPU monitoring with a longer interval
        psutil.cpu_percent(interval=None, percpu=False)
        self.process.cpu_percent()
        
    def get_stats(self):
        """Get current CPU, RAM, and GPU statistics."""
        # Use interval=None to get non-blocking CPU measurement
        cpu_percent = self.process.cpu_percent()
        
        # Also get system-wide CPU to ensure we capture usage
        sys_cpu = psutil.cpu_percent(interval=None)
        cpu_percent = max(cpu_percent, sys_cpu)
        
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        mem_percent = self.process.memory_percent()
        
        # Get GPU stats if available
        gpu_util = 0.0
        gpu_mem_mb = 0.0
        if self.has_gpu and self.gpu_handles:
            try:
                # Average across all GPUs
                total_util = 0
                total_mem = 0
                for handle in self.gpu_handles:
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    total_util += util.gpu
                    mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mem += mem_info.used / 1024 / 1024  # Convert to MB
                
                gpu_util = total_util / len(self.gpu_handles)
                gpu_mem_mb = total_mem / len(self.gpu_handles)
            except Exception:
                pass
        
        # Track maximums
        self.max_cpu = max(self.max_cpu, cpu_percent)
        self.max_mem_mb = max(self.max_mem_mb, mem_mb)
        self.max_mem_pct = max(self.max_mem_pct, mem_percent)
        self.max_gpu_util = max(self.max_gpu_util, gpu_util)
        self.max_gpu_mem_mb = max(self.max_gpu_mem_mb, gpu_mem_mb)
        
        return cpu_percent, mem_mb, mem_percent, gpu_util, gpu_mem_mb
    
    def cleanup(self):
        """Cleanup GPU monitoring resources."""
        if self.has_gpu:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


def simple_sieve(limit, show_progress=False):
    """
    Generate all primes up to limit using simple sieve.
    Used to find base primes for segmented sieve.
    
    Args:
        limit: Upper limit for prime generation
        show_progress: Whether to show progress message
        
    Returns:
        List of primes up to limit
    """
    if limit < 2:
        return []
    
    if show_progress:
        console.print(f"[dim]Computing base primes up to {limit:,}...[/dim]")
    
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]


def segmented_sieve(start, end, progress=None, task_id=None, monitor=None, largest_prime_ref=None):
    """
    Find all prime numbers in the range [start, end] using segmented sieve.
    This is memory-efficient for very large ranges.
    
    Args:
        start: Starting number (inclusive)
        end: Ending number (inclusive)
        progress: Rich Progress object (optional)
        task_id: Task ID for progress tracking (optional)
        monitor: SystemMonitor object for resource tracking (optional)
        largest_prime_ref: List to store largest prime found so far (optional)
    
    Yields:
        Prime numbers in the range
    """
    if end < 2:
        return
    
    start = max(2, start)
    
    # Segment size (balance between memory and performance)
    # Use 1M as segment size for good performance
    segment_size = 1000000
    
    # Get base primes up to sqrt(end)
    limit = int(math.sqrt(end)) + 1
    base_primes = simple_sieve(limit, show_progress=True)
    
    if progress and task_id is not None:
        console.print(f"[dim]Found {len(base_primes):,} base primes. Starting segmented sieve...[/dim]")
    
    # If start is within the base primes range, yield those first
    if start <= limit:
        for prime in base_primes:
            if start <= prime <= end:
                if largest_prime_ref is not None:
                    largest_prime_ref[0] = prime
                yield prime
        start = limit + 1
        if start > end:
            return
    
    # Process segments
    total_segments = math.ceil((end - start + 1) / segment_size)
    
    for seg_idx in range(total_segments):
        low = start + seg_idx * segment_size
        high = min(low + segment_size - 1, end)
        
        # Create a boolean array for this segment
        seg_size = high - low + 1
        is_prime = [True] * seg_size
        
        # Use base primes to mark composites in this segment
        for prime in base_primes:
            # Find the first multiple of prime in [low, high]
            first_multiple = ((low + prime - 1) // prime) * prime
            
            # Make sure we don't mark the prime itself as composite
            if first_multiple == prime:
                first_multiple += prime
            
            # Mark all multiples as composite
            for j in range(first_multiple, high + 1, prime):
                is_prime[j - low] = False
        
        # Yield primes in this segment
        for i in range(seg_size):
            if is_prime[i]:
                prime = low + i
                if largest_prime_ref is not None:
                    largest_prime_ref[0] = prime
                yield prime
        
        # Update progress after processing each segment
        if progress and task_id is not None and monitor:
            cpu, mem_mb, mem_pct, gpu_util, gpu_mem_mb = monitor.get_stats()
            
            # Build description with proper formatting
            desc_parts = [
                f"[cyan]Segment {seg_idx + 1}/{total_segments}[/cyan]",
                f"[yellow]CPU: {cpu:5.1f}%[/yellow]",
                f"[green]RAM: {mem_mb:7.1f}MB ({mem_pct:4.1f}%)[/green]"
            ]
            
            if monitor.has_gpu:
                desc_parts.append(f"[magenta]GPU: {gpu_util:5.1f}% ({gpu_mem_mb:7.1f}MB)[/magenta]")
            
            if largest_prime_ref is not None and largest_prime_ref[0] > 0:
                desc_parts.append(f"[blue]Largest: {largest_prime_ref[0]:,}[/blue]")
            
            progress.update(
                task_id,
                advance=1,
                description=" | ".join(desc_parts)
            )


def sieve_of_eratosthenes(start, end, progress=None, task_id=None, monitor=None, largest_prime_ref=None):
    """
    Find all prime numbers in the range [start, end] using the Sieve of Eratosthenes.
    Automatically chooses between simple and segmented sieve based on range size.
    
    Args:
        start: Starting number (inclusive)
        end: Ending number (inclusive)
        progress: Rich Progress object (optional)
        task_id: Task ID for progress tracking (optional)
        monitor: SystemMonitor object for resource tracking (optional)
        largest_prime_ref: List to store largest prime found so far (optional)
    
    Yields:
        Prime numbers in the range
    """
    if end < 2:
        return
    
    # Adjust start to be at least 2
    start = max(2, start)
    
    # For large ranges (> 100 million), use segmented sieve
    # This prevents memory issues
    if end > 100_000_000:
        yield from segmented_sieve(start, end, progress, task_id, monitor, largest_prime_ref)
        return
    
    # Create a boolean array and initialize all entries as true
    # We only need to track numbers from start to end
    # But for efficiency, we sieve from 2 to end
    size = end + 1
    is_prime = [True] * size
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes with progress bar
    sqrt_end = int(math.sqrt(end)) + 1
    
    for i in range(2, sqrt_end):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i * i, size, i):
                is_prime[j] = False
        
        # Update progress with system stats
        if progress and task_id is not None and monitor:
            cpu, mem_mb, mem_pct, gpu_util, gpu_mem_mb = monitor.get_stats()
            
            desc_parts = [
                f"[cyan]Sieving[/cyan]",
                f"[yellow]CPU: {cpu:5.1f}%[/yellow]",
                f"[green]RAM: {mem_mb:7.1f}MB ({mem_pct:4.1f}%)[/green]"
            ]
            
            if monitor.has_gpu:
                desc_parts.append(f"[magenta]GPU: {gpu_util:5.1f}% ({gpu_mem_mb:7.1f}MB)[/magenta]")
            
            progress.update(
                task_id,
                advance=1,
                description=" | ".join(desc_parts)
            )
    
    # Yield primes in the requested range
    for num in range(start, end + 1):
        if is_prime[num]:
            if largest_prime_ref is not None:
                largest_prime_ref[0] = num
            yield num


def get_range_from_user():
    """
    Get the search range from user input.
    
    Returns:
        Tuple of (start, end, save_to_file) for the range and save option
    """
    console.print()
    title = Text("PRIME NUMBER SEARCHER", style="bold magenta")
    console.print(Panel(title, border_style="bright_blue"))
    console.print()
    console.print("[cyan]Find all prime numbers in a specified range.[/cyan]")
    console.print("[cyan]Using the Sieve of Eratosthenes algorithm for maximum efficiency.[/cyan]")
    console.print()
    
    while True:
        try:
            start = int(console.input("[yellow]Enter the starting number (min 1): [/yellow]"))
            if start < 1:
                console.print("[red]Starting number must be at least 1. Please try again.[/red]")
                continue
            break
        except ValueError:
            console.print("[red]Invalid input. Please enter a valid integer.[/red]")
    
    while True:
        try:
            end = int(console.input(f"[yellow]Enter the ending number (min {start}): [/yellow]"))
            if end < start:
                console.print(f"[red]Ending number must be at least {start}. Please try again.[/red]")
                continue
            break
        except ValueError:
            console.print("[red]Invalid input. Please enter a valid integer.[/red]")
    
    # Ask if user wants to save to file
    while True:
        save_input = console.input("[yellow]Save primes to file? (y/n): [/yellow]").strip().lower()
        if save_input in ['y', 'yes']:
            save_to_file = True
            break
        elif save_input in ['n', 'no']:
            save_to_file = False
            break
        else:
            console.print("[red]Please enter 'y' or 'n'.[/red]")
    
    return start, end, save_to_file


def save_primes_to_file(primes, filename="primes.txt"):
    """
    Save prime numbers to a text file, one per line.
    
    Args:
        primes: List of prime numbers
        filename: Name of the output file
    """
    with open(filename, 'w') as f:
        for prime in primes:
            f.write(f"{prime}\n")


def main():
    """Main function to run the prime searcher."""
    # Get search range from user
    start, end, save_to_file = get_range_from_user()
    
    console.print()
    console.print(f"[cyan]Searching for primes in range [[/cyan][bold]{start:,}[/bold][cyan], [/cyan][bold]{end:,}[/bold][cyan]]...[/cyan]")
    console.print("[dim]This may take a moment for large ranges.[/dim]")
    
    # Determine if using segmented sieve
    if end > 100_000_000:
        console.print(f"[yellow]⚡[/yellow] Using segmented sieve for large range (memory efficient)")
    console.print()
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    if monitor.has_gpu:
        console.print("[dim]GPU monitoring enabled[/dim]")
    
    # Find primes with progress bar
    primes = []
    largest_prime = [0]  # Use list to allow modification in nested function
    
    # Determine total steps based on algorithm
    if end > 100_000_000:
        # Segmented sieve - count segments
        segment_size = 1000000
        limit = int(math.sqrt(end)) + 1
        adjusted_start = max(start, limit + 1)
        if adjusted_start <= end:
            total_steps = math.ceil((end - adjusted_start + 1) / segment_size)
        else:
            total_steps = 1
    else:
        # Simple sieve - count up to sqrt
        sqrt_end = int(math.sqrt(end)) + 1
        total_steps = sqrt_end - 2
    
    # Create custom progress bar with CPU/RAM/GPU monitoring
    with Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="bright_blue", complete_style="bright_green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,  # Update more frequently to catch CPU usage
    ) as progress:
        
        initial_desc = "[cyan]Sieving[/cyan] | [yellow]CPU: 0.0%[/yellow] | [green]RAM: 0.0MB (0.0%)[/green]"
        if monitor.has_gpu:
            initial_desc += " | [magenta]GPU: 0.0% (0.0MB)[/magenta]"
        
        task = progress.add_task(
            initial_desc,
            total=total_steps
        )
        
        # Run the sieve
        start_time = time.time()
        for prime in sieve_of_eratosthenes(start, end, progress, task, monitor, largest_prime):
            primes.append(prime)
        
        elapsed_time = time.time() - start_time
    
    # Display results
    console.print()
    results_text = Text("RESULTS", style="bold green")
    console.print(Panel(results_text, border_style="bright_green"))
    console.print()
    console.print(f"[green]✓[/green] Found [bold]{len(primes):,}[/bold] prime numbers in range [[bold]{start:,}[/bold], [bold]{end:,}[/bold]]")
    console.print(f"[cyan]⏱[/cyan]  Completed in [bold]{elapsed_time:.2f}[/bold] seconds")
    
    # Show peak stats
    console.print(f"[yellow]📊[/yellow] Peak CPU usage: [bold]{monitor.max_cpu:.1f}%[/bold]")
    console.print(f"[yellow]📊[/yellow] Peak RAM usage: [bold]{monitor.max_mem_mb:.1f} MB[/bold] ([bold]{monitor.max_mem_pct:.1f}%[/bold])")
    
    if monitor.has_gpu:
        console.print(f"[magenta]📊[/magenta] Peak GPU usage: [bold]{monitor.max_gpu_util:.1f}%[/bold] ([bold]{monitor.max_gpu_mem_mb:.1f} MB[/bold])")
    
    if largest_prime[0] > 0:
        console.print(f"[blue]🔢[/blue] Largest prime found: [bold]{largest_prime[0]:,}[/bold]")
    
    # Save to file if requested
    if save_to_file:
        save_primes_to_file(primes)
        console.print(f"\n[green]💾[/green] Primes saved to [bold cyan]'primes.txt'[/bold cyan]")
    else:
        console.print(f"\n[dim]💾 File saving skipped (as requested)[/dim]")
    
    # Show first and last few primes if there are many
    if len(primes) > 10:
        console.print(f"\n[dim]First 5 primes:[/dim] [yellow]{primes[:5]}[/yellow]")
        console.print(f"[dim]Last 5 primes:[/dim] [yellow]{primes[-5:]}[/yellow]")
    elif len(primes) > 0:
        console.print(f"\n[dim]Primes found:[/dim] [yellow]{primes}[/yellow]")
    
    console.print()
    
    # Cleanup
    monitor.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[red]Operation cancelled by user.[/red]")
        sys.exit(0)
