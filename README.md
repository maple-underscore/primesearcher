# primesearcher

A Python-based toolkit for prime number searching and mathematical conjecture exploration.

## Features

### 1. Prime Number Searcher (`prime_searcher.py`)
- **Efficient Algorithm**: Uses the Sieve of Eratosthenes for optimal performance
- **Range Selection**: Interactive UI to specify custom search ranges
- **Progress Visualization**: Real-time progress bar during computation
- **File Output**: Saves all found primes to `primes.txt` (one per line)

### 2. 3n-1 Conjecture Evaluator (`conjecture_3n_minus_1.py`)
- **Mathematical Exploration**: Tests numbers against the 3n-1 conjecture
- **Visual Graphs**: Generates matplotlib graphs for each number's sequence
- **Multiple Input Modes**: Single number, range, or custom list
- **Detailed Results**: Shows steps, max values, and convergence status

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Prime Number Searcher

```bash
python3 prime_searcher.py
```

Follow the interactive prompts to:
1. Enter a starting number (minimum 1)
2. Enter an ending number
3. View the progress bar as primes are found
4. Results are saved to `primes.txt`

Example:
```
Enter the starting number (min 1): 1
Enter the ending number (min 1): 1000

Searching for primes in range [1, 1000]...
Found 168 prime numbers
Primes saved to 'primes.txt'
```

### 3n-1 Conjecture Evaluator

```bash
python3 conjecture_3n_minus_1.py
```

The 3n-1 conjecture applies these rules iteratively:
- If n is even: n → n/2
- If n is odd: n → 3n-1

Choose from three input modes:
1. **Single number**: Evaluate one specific number
2. **Range of numbers**: Evaluate all numbers in a range
3. **Custom list**: Evaluate a comma-separated list of numbers

Graphs are saved to the `conjecture_graphs/` directory.

Example:
```
Choose input mode:
  1. Single number
  2. Range of numbers
  3. Custom list
Enter choice (1-3): 3
Enter numbers separated by commas: 5,12,27,100
```

## Output Files

- `primes.txt`: Contains all found prime numbers (one per line)
- `conjecture_graphs/`: Directory containing PNG graphs for each evaluated number

## Requirements

- Python 3.7+
- tqdm (progress bars)
- matplotlib (graph generation)
- numpy (numerical computations)

## License

Open source - free to use and modify.