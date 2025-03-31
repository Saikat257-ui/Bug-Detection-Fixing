def calculate_total(numbers):
    total = 0
    for i in range(len(numbers) + 1):  # Off-by-one error
        total += numbers[i]  # Index out of range for the last iteration
    return totl  # Typo: 'totl' instead of 'total'

def reverse_string(s):
    reversed_str = ''
    for i in range(len(s)):  
        reversed_str += s[len(s) - i]  # Index error: accessing out of bounds
    return reversed_str

def fetch_data_from_dict(data_dict, key):
    return data_dict.key  # Incorrect attribute access: should use square brackets []

nums = [1, 2, 3, 4, 5]
total_sum = calculate_total(nums)
print(tota_sum)  # Typo: 'tota_sum' instead of 'total_sum'

string = "debug"
reversed_result = reverse_string(string)
print(reversed_results)  # Typo: 'reversed_results' instead of 'reversed_result'

data = {"name": "AI", "age": 3}
age = fetch_data_from_dict(data, "age")
print(ag)  # Typo: 'ag' instead of 'age'
