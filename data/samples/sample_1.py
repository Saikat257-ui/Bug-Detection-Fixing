def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    

data = [1, 2, 3, 4, 5]
avg = calculate_average(data)
print(f"The average is: {avg}")

