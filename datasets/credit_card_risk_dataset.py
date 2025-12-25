from datasets import load_dataset

ds = load_dataset("saifhmb/CreditCardRisk")

# """For each instance, there is an integer for the ID , an integer for the age, an integer for the income, a string for the gender with 2 possible values m for male and f for female, a string for the marital status with 3 possible values: married, single, divsepwid (represents divorced, separated, widow), an integer for the numkids, an integer for the numcards, a string for the howpaid with 2 possible values, weekly or monthly, a string for the mortgage with 2 possible values y (yes) or n (no), an integer for the storecar, an integer for the loans, and a string for the risk with 3 possible values, bad profit, bad loss, or good risk.

# {'ID': '100,756', 'AGE': '44', 'INCOME': '59,944', 'GENDER': 'm', 'MARITAL': 'married', 'NUMKIDS': '1', 'NUMCARDS': '2', 'HOWPAID': 'monthly', 'MORTGAGE': 'y', 'STORECAR': '2', 'LOANS': '0', 'RISK': 'good risk', }

# Data Fields
# ID: an integer with a unique ID for each customer
# AGE: an integer stating the age of the customer
# INCOME: an integer stating the income of the customer in USD
# GENDER: a string stating the gender of the customer with 2 possible values, either m (male) or f (female)
# MARITAL: a string stating the marital status of the customer 3 possible values, either married, single, or divsepwid
# NUMKIDS: an integer stating the number of children each customer has
# NUMCARDS: an integer stating the number of cards each customer has
# HOWPAID: a string stating the frequency of payment received by each customer with 2 possible values, monthly or weekly
# MORTGAGE: a string stating whether a customer has mortgage with 2 possible values, y or no
# STORECAR: an integer stating the number of store credit cards each customer has
# LOANS: an integer stating the number of outstanding loans each customer has
# RISK: a string stating the credit card risk per customer with 3 possible values, bad loss, bad profit or good risk"""