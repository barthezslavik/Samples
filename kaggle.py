import sqlite3
import csv

def convert_to_csv(db_file, output_dir):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name from sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name in tables:
        table_name = table_name[0]
        csv_file = f'{output_dir}/{table_name}.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([i[0] for i in cursor.description])
            cursor.execute(f"SELECT * from {table_name}")
            writer.writerows(cursor)

if __name__ == '__main__':
    db_file = 'database.db'
    output_dir = 'output'
    convert_to_csv(db_file, output_dir)
