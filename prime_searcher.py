#!/usr/bin/env python3
"""
Prime Number Searcher

An efficient prime number search tool using the Sieve of Eratosthenes algorithm.
Saves prime numbers to primes.txt with an intuitive progress bar.
"""

import sys
from tqdm import tqdm


def sieve_of_eratosthenes(start, end):
    """
    Find all prime numbers in the range [start, end] using the Sieve of Eratosthenes.
    
    Args:
        start: Starting number (inclusive)
        end: Ending number (inclusive)
    
    Yields:
        Prime numbers in the range
    """
    if end < 2:
        return
    
    # Adjust start to be at least 2
    start = max(2, start)
    
    # Create a boolean array and initialize all entries as true
    # We only need to track numbers from start to end
    # But for efficiency, we sieve from 2 to end
    size = end + 1
    is_prime = [True] * size
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes with progress bar
    sqrt_end = int(end ** 0.5) + 1
    
    with tqdm(total=sqrt_end-2, desc="Sieving", unit="num", leave=False) as pbar:
        for i in range(2, sqrt_end):
            if is_prime[i]:
                # Mark multiples of i as not prime
                for j in range(i * i, size, i):
                    is_prime[j] = False
            pbar.update(1)
    
    # Yield primes in the requested range
    for num in range(start, end + 1):
        if is_prime[num]:
            yield num


def get_range_from_user():
    """
    Get the search range from user input.
    
    Returns:
        Tuple of (start, end) for the range
    """
    print("\n" + "="*50)
    print("PRIME NUMBER SEARCHER")
    print("="*50)
    print("\nFind all prime numbers in a specified range.")
    print("Using the Sieve of Eratosthenes algorithm for maximum efficiency.\n")
    
    while True:
        try:
            start = int(input("Enter the starting number (min 1): "))
            if start < 1:
                print("Starting number must be at least 1. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    
    while True:
        try:
            end = int(input(f"Enter the ending number (min {start}): "))
            if end < start:
                print(f"Ending number must be at least {start}. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    
    return start, end


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
    start, end = get_range_from_user()
    
    print(f"\nSearching for primes in range [{start}, {end}]...")
    print("This may take a moment for large ranges.\n")
    
    # Find primes with progress bar
    primes = []
    
    # First, run the sieve
    for prime in sieve_of_eratosthenes(start, end):
        primes.append(prime)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Found {len(primes)} prime numbers in range [{start}, {end}]")
    
    # Save to file
    save_primes_to_file(primes)
    print(f"\nPrimes saved to 'primes.txt'")
    
    # Show first and last few primes if there are many
    if len(primes) > 10:
        print(f"\nFirst 5 primes: {primes[:5]}")
        print(f"Last 5 primes: {primes[-5:]}")
    else:
        print(f"\nPrimes found: {primes}")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
