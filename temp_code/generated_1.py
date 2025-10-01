def print_fibonacci_numbers(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

# Call the function to print the first 10 Fibonacci numbers
print_fibonacci_numbers(10)