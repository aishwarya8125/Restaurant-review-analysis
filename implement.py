import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk, messagebox
import sqlite3

# Load dataset
dataset = pd.read_csv('c:/Users/AISHWARYA/Desktop/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
rras_code = "Wyd^H3R"
food_rev = {}
food_perc = {}
food_prices = {
    "Idly": 50, "Dosa": 80, "Vada": 40, "Roti": 20, "Meals": 150,
    "Veg Biryani": 120, "Egg Biryani": 150, "Chicken Biryani": 200,
    "Mutton Biryani": 250, "Ice Cream": 60, "Noodles": 70,
    "Manchurian": 100, "Orange juice": 30, "Apple Juice": 30,
    "Pineapple juice": 30, "Banana juice": 30
}

conn = sqlite3.connect('Restaurant_food_data.db')
c = conn.cursor()

# Preprocess reviews
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Prepare data for model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Food items
foods = list(food_prices.keys())

for i in foods:
    food_rev[i] = []
    food_perc[i] = [0.0, 0.0]

def init_data():
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute("DELETE FROM item")
    for i in range(len(foods)):
        c.execute("INSERT INTO item VALUES(:item_name,:no_of_customers,\
        :no_of_positives,:no_of_negatives,:pos_perc,:neg_perc)",
                  {
                      'item_name': foods[i],
                      'no_of_customers': "0",
                      'no_of_positives': "0",
                      'no_of_negatives': "0",
                      'pos_perc': "0.0%",
                      'neg_perc': "0.0%"
                  }
                  )
    conn.commit()
    conn.close()

# Initialize the GUI
root1 = Tk()
main = "Restaurant Review Analysis System/"
root1.title(main + "Welcome Page")

label = Label(root1, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
              bd=2, font=('Arial', 47, 'bold', 'underline'))

ques = Label(root1, text="Are you a Customer or Owner ???")

cust = Button(root1, text="Customer", font=('Arial', 20),
              padx=80, pady=20, command=lambda: take_review())

owner = Button(root1, text="Owner", font=('Arial', 20),
              padx=100, pady=20, command=lambda: login())

root1.state('zoomed')
label.grid(row=0, column=0)
ques.grid(row=1, column=0, sticky=W + E)
ques.config(font=("Helvetica", 30))
cust.grid(row=2, column=0)
owner.grid(row=3, column=0)

def take_review():
    root2 = Toplevel()
    root2.title(main + "Give Review")

    label = Label(root2, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                  bd=2, font=('Arial', 47, 'bold', 'underline'))

    req1 = Label(root2, text="Select the item(s) you have taken.....")

    chk_btns = []
    selected_foods = []
    req2 = Label(root2, text="Give your review below....")
    rev_tf = Entry(root2, width=125, borderwidth=5)
    global variables
    variables = []

    for i in range(len(foods)):
        var = IntVar()
        chk = Checkbutton(root2, text=foods[i], variable=var)
        variables.append(var)
        chk_btns.append(chk)

    label.grid(row=0, column=0, columnspan=4)
    req1.grid(row=1, column=0, columnspan=4, sticky=W + E)
    req1.config(font=("Helvetica", 30))

    for i in range(4):
        for j in range(4):
            c = chk_btns[i * 4 + j]
            c.grid(row=i + 3, column=j, columnspan=1, sticky=W)

    submit_review = Button(root2, text="Submit Review", font=(
        'Arial', 20), padx=100, pady=20, command=lambda: [
        estimate(rev_tf.get()), root2.destroy()])

    root2.state('zoomed')
    req2.grid(row=7, column=0, columnspan=4, sticky=W + E)
    req2.config(font=("Helvetica", 20))
    rev_tf.grid(row=8, column=1, rowspan=3, columnspan=2, sticky=S)
    submit_review.grid(row=12, column=0, columnspan=4)

def estimate(s):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    review = re.sub('[^a-zA-Z]', ' ', s)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    X = cv.transform([review]).toarray()
    res = classifier.predict(X)

    if "not" in review:
        res[0] = abs(res[0] - 1)

    selected_foods = []
    for i in range(len(foods)):
        if variables[i].get() == 1:
            selected_foods.append(foods[i])

    c.execute("SELECT *,oid FROM item")
    records = c.fetchall()

    for i in records:
        rec = list(i)
        if rec[0] in selected_foods:
            n_cust = int(rec[1]) + 1
            n_pos = int(rec[2])
            n_neg = int(rec[3])

            if res[0] == 1:
                n_pos += 1
            else:
                n_neg += 1

            pos_percent = round((n_pos / n_cust) * 100, 1)
            neg_percent = round((n_neg / n_cust) * 100, 1)
            c.execute("""UPDATE item SET Item_name=:item_name,No_of_customers\
            =:no_of_customers,No_of_positive_reviews=:no_of_positives,\
            No_of_negative_reviews=:no_of_negatives,Positive_percentage\
            =:pos_perc,Negative_percentage=:neg_perc where oid=:Oid""",
                      {
                          'item_name': rec[0],
                          'no_of_customers': str(n_cust),
                          'no_of_positives': str(n_pos),
                          'no_of_negatives': str(n_neg),
                          'pos_perc': str(pos_percent) + "%",
                          'neg_perc': str(neg_percent) + "%",
                          'Oid': foods.index(rec[0]) + 1
                      }
                      )

    conn.commit()
    conn.close()

    # Display the bill with costs and tax
    display_bill(selected_foods)

def display_bill(selected_foods):
    bill_window = Toplevel()
    bill_window.title("Bill Details")
    
    # Create a treeview for displaying the bill
    bill_tree = ttk.Treeview(bill_window, columns=('Item', 'Price'), show='headings')
    bill_tree.heading('Item', text='Item')
    bill_tree.heading('Price', text='Price')
    bill_tree.pack(pady=10)

    total_cost = 0
    tax_rate = 0.05  # 5% tax
    for food in selected_foods:
        cost = food_prices[food]
        total_cost += cost
        bill_tree.insert('', 'end', values=(food, f"₹{cost}"))

    total_tax = total_cost * tax_rate
    total_amount = total_cost + total_tax

    # Display total cost and tax
    total_label = Label(bill_window, text=f"Total Cost: ₹{total_cost:.2f}\nTax (5%): ₹{total_tax:.2f}\nTotal Amount: ₹{total_amount:.2f}")
    total_label.pack(pady=10)

def login():
    def verify():
        username = user_entry.get()
        password = pass_entry.get()

        if username == "admin" and password == "admin123":
            root3.destroy()
            owner_page()
        else:
            messagebox.showerror("Error", "Invalid Credentials")

    root3 = Toplevel()
    root3.title(main + "Owner Login")

    user_label = Label(root3, text="Username:")
    user_entry = Entry(root3)
    pass_label = Label(root3, text="Password:")
    pass_entry = Entry(root3, show="*")

    login_btn = Button(root3, text="Login", command=verify)
    
    user_label.grid(row=0, column=0, padx=10, pady=10)
    user_entry.grid(row=0, column=1, padx=10, pady=10)
    pass_label.grid(row=1, column=0, padx=10, pady=10)
    pass_entry.grid(row=1, column=1, padx=10, pady=10)
    login_btn.grid(row=2, columnspan=2, padx=10, pady=10)

def owner_page():
    root4 = Toplevel()
    root4.title(main + "Owner Page")

    label = Label(root4, text="RESTAURANT REVIEW ANALYSIS SYSTEM", bd=2, font=('Arial', 47, 'bold', 'underline'))
    label.grid(row=0, column=0, columnspan=2)

    clear_data_btn = Button(root4, text="Clear All Reviews", font=('Arial', 20),
                             padx=50, pady=20, command=clear_data)
    clear_data_btn.grid(row=1, column=0, columnspan=2)

    view_data_btn = Button(root4, text="View Reviews", font=('Arial', 20),
                            padx=50, pady=20, command=view_data)
    view_data_btn.grid(row=2, column=0, columnspan=2)

    init_data()  # Initialize the data if necessary

    root4.state('zoomed')

def clear_data():
    confirm = messagebox.askyesno("Confirm", "Are you sure you want to clear all reviews?")
    if confirm:
        conn = sqlite3.connect('Restaurant_food_data.db')
        c = conn.cursor()
        c.execute("DELETE FROM item")  # Delete all reviews from the table
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", "All reviews cleared successfully!")

def view_data():
    view_window = Toplevel()
    view_window.title("View Reviews")
    
    # Create a treeview for displaying the reviews
    review_tree = ttk.Treeview(view_window, columns=('Item', 'Customers', 'Positives', 'Negatives', 'Pos %', 'Neg %'), show='headings')
    review_tree.heading('Item', text='Item')
    review_tree.heading('Customers', text='No. of Customers')
    review_tree.heading('Positives', text='No. of Positives')
    review_tree.heading('Negatives', text='No. of Negatives')
    review_tree.heading('Pos %', text='Positive %')
    review_tree.heading('Neg %', text='Negative %')
    review_tree.pack(pady=10)

    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM item")
    records = c.fetchall()

    for record in records:
        review_tree.insert('', 'end', values=record)

    conn.close()

# Run the application
root1.mainloop()

    




