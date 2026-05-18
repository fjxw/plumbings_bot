import sqlite3
import os

DB_PATH = 'bot_database.db'

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Таблица товаров
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        name TEXT NOT NULL,
        price INTEGER NOT NULL,
        stock INTEGER DEFAULT 0
    )
    ''')
    
    # Таблица заказов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        items TEXT NOT NULL,
        total_price INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def migrate_from_json(products_json):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Проверяем, есть ли уже товары
    cursor.execute('SELECT COUNT(*) FROM products')
    if cursor.fetchone()[0] == 0:
        for category, items in products_json.items():
            for item in items:
                cursor.execute(
                    'INSERT INTO products (category, name, price, stock) VALUES (?, ?, ?, ?)',
                    (category, item['name'], item['price'], 10) # По умолчанию 10 шт
                )
    
    conn.commit()
    conn.close()

def get_categories():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT category FROM products')
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return categories

def get_products_by_category(category, max_price=None):
    conn = get_connection()
    cursor = conn.cursor()
    if max_price:
        cursor.execute('SELECT name, price, stock FROM products WHERE category = ? AND price <= ? AND stock > 0', (category, max_price))
    else:
        cursor.execute('SELECT name, price, stock FROM products WHERE category = ? AND stock > 0', (category,))
    
    products = [{'name': row[0], 'price': row[1], 'stock': row[2]} for row in cursor.fetchall()]
    conn.close()
    return products

def add_product(category, name, price, stock):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO products (category, name, price, stock) VALUES (?, ?, ?, ?)', (category, name, price, stock))
    conn.commit()
    conn.close()

def delete_product(product_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM products WHERE name = ?', (product_name,))
    conn.commit()
    conn.close()

def update_stock(product_name, amount):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE products SET stock = stock - ? WHERE name = ?', (amount, product_name))
    conn.commit()
    conn.close()

def set_stock(product_name, stock):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE products SET stock = ? WHERE name = ?', (stock, product_name))
    conn.commit()
    conn.close()

def save_order(order_id, user_id, items, total_price):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO orders (id, user_id, items, total_price) VALUES (?, ?, ?, ?)', 
                   (order_id, user_id, str(items), total_price))
    conn.commit()
    conn.close()

def get_all_orders():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM orders ORDER BY created_at DESC')
    orders = cursor.fetchall()
    conn.close()
    return orders

def search_products(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT category, name, price, stock FROM products WHERE name LIKE ? AND stock > 0', ('%' + query + '%',))
    products = [{'category': row[0], 'name': row[1], 'price': row[2], 'stock': row[3]} for row in cursor.fetchall()]
    conn.close()
    return products

def get_top_products():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT category, name, price, stock FROM products ORDER BY stock DESC LIMIT 5')
    products = [{'category': row[0], 'name': row[1], 'price': row[2], 'stock': row[3]} for row in cursor.fetchall()]
    conn.close()
    return products
