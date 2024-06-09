import os
import psycopg2
def get_reviewer_id(user_id):
    connection_string = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    # Query to retrieve reviewer_id based on user_id
    query = """
    SELECT user_id
    FROM books_recommendation.reviews
    WHERE user_id = %s
    """
    cur.execute(query, (user_id,))
    reviewer_id = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return reviewer_id[0] if reviewer_id else None