import os
import psycopg2
from dotenv import load_dotenv


# app = Flask(__name__)
load_dotenv()

# Function to retrieve book data from the database
def get_book_data():
    connection_string = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    # Query to retrieve book details
    query = """
    SELECT b.book_title, bd.average_rating, bd.author_name, bd.book_image_url,bd.total_book_reviews
    FROM books_recommendation.books b
    JOIN books_recommendation.book_details bd ON b.book_id = bd.book_id
    ORDER BY bd.total_book_reviews DESC
    """
    cur.execute(query)
    books = cur.fetchall()
    
    # Convert to a list of dictionaries
    book_list = []
    for book in books:
        book_dict = {
            "Title": book[0],
            "Rating": book[1],
            "Author": book[2],
            "ImageURL": book[3],
            "NumReviews": book[4]
        }
        book_list.append(book_dict)
    
    cur.close()
    conn.close()
    
    return book_list
