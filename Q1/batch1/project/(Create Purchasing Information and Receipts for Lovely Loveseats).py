#lovely lovesea descriotion
lovely_loveseat_description = """ Lovely Loveseat. Tufted polyester blend on wood. 32 inches high x 40 inches wide x inches deep. Red or white"""
lovely_loveseat_price = 254.00

#Stylish settee
stylish_sette_description =  """ Stylish settee. Faux leaather on brich. 29.50 inches high x 54.75 inches wide x 28 inches deep. Black."""
Stylist_settee_price = 180.50

#luxurious lamp
luxurious_lamp_description = """ Luxurious Lamp. Glass and iron.36 inches tall.brown with cream shade."""
luxurious_lamp_price = 52.15

#sale tax 
sales_tax = 0.88

customer_one_total = 0
customer_one_itemization = ""


# customer one shoppping 
customer_one_total += lovely_loveseat_price
customer_one_itemization = lovely_loveseat_description

customer_one_total += luxurious_lamp_price

customer_one_itemization = "\n"+ lovely_loveseat_description + "\n " + luxurious_lamp_description

customer_one_tax = customer_one_total * sales_tax
customer_one_total += customer_one_tax

print("Customer One Items:" + customer_one_itemization )

print("Customer One Total:"+ str (customer_one_total))
