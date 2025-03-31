def calculate_discounted_price(price, discount_rate):
    discount = price * discount_rate
    final_price = price - discount
    return "The final price is " + final_price

# Bug: Type error - string concatenation with number
