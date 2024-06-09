from application.Model import user_profiling
import os 
import psycopg2

def get_recommendation(user_id=None, book_title=None, top_n=10):
    profiling = user_profiling.UserProfiling()
    top_n += 5  # To get a few extra recommendations

    connection_string = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    schema_name = 'books_recommendation'
    table_name_name = 'books'

    book_id = None
    if book_title:
        cur.execute(f"SELECT book_id FROM {schema_name}.{table_name_name} WHERE book_title = %s", (book_title,))
        book_id_row = cur.fetchone()
        if book_id_row:
            book_id = book_id_row[0]
        else:
            cur.close()
            conn.close()
            return [], []

    recommendations, scores = profiling.recommend_books(user_id, book_id, top_n)
    if recommendations is None:
        cur.close()
        conn.close()
        return [], []

    book_details = []
    for i in list(recommendations):
        cur.execute(f"""
            SELECT 
                b.book_title, 
                bd.author_name, 
                bd.average_rating, 
                bd.genres, 
                bd.book_image_url,
                bd.total_book_reviews
            FROM {schema_name}.books b
            JOIN {schema_name}.book_details bd ON b.book_id = bd.book_id
            LEFT JOIN {schema_name}.reviews r ON b.book_id = r.book_id
            WHERE b.book_id = %s
        """, (i,))
        detail_row = cur.fetchone()
        if detail_row:
            book_details.append({
                "Title": detail_row[0],
                "Author": detail_row[1],
                "Rating": round(detail_row[2], 2) if detail_row[2] is not None else "N/A",
                "Genre": detail_row[3].strip("[]").replace("'", ""),
                "ImageURL": detail_row[4],
                "NumReviews": detail_row[5]
            })

    cur.close()
    conn.close()

    return book_details[1:top_n-5], scores[1:top_n-5]
