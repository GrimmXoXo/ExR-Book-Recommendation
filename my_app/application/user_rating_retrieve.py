import os
import psycopg2

def fetch_user_ratings(user_id):
    connection_string = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    query = """
    SELECT b.book_title, r.rating, bd.author_name, bd.genres, bd.book_image_url
    FROM books_recommendation.reviews r
    JOIN books_recommendation.books b ON r.book_id = b.book_id
    JOIN books_recommendation.book_details bd ON b.book_id = bd.book_id
    WHERE r.user_id = %s
    """
    cur.execute(query, (user_id,))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return results