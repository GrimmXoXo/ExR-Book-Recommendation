from application import app
from flask import render_template, url_for, request, redirect, flash, session
# from application.models import , users
from application.forms import LoginForm
from application.models import get_book_data
from application.user_id_retrieve import get_reviewer_id
from application.user_rating_retrieve import fetch_user_ratings
from application.Model.model import get_recommendation



@app.route("/")
def index():
    return render_template("index.html", navindex=True)

@app.route("/catalog")
def catalog():
    book_data = get_book_data() 
    return render_template("catalog.html", bookcat=book_data)#moviecat=moviecat

@app.route("/shelve")
def shelve():
    if not session.get('user_id'):
        flash("Please login to access shelve", "danger")
        print("DEBUG: Access denied to shelve. No user_id in session.")
        return redirect(url_for('login'))  # If not logged in, redirect user to login page
    
    user_id = session['user_id']
    ratings = fetch_user_ratings(user_id)
    
    # Transform the results into a list of dictionaries
    ratings_dict = []
    for r in ratings:
        try:
            title = r[0]
        except IndexError:
            title = None
        
        try:
            rating = r[1]
        except IndexError:
            rating = None
        
        try:
            author = r[2]
        except IndexError:
            author = None
        
        try:
            genre = r[3].strip("[]").replace("'", "")
        except (IndexError, AttributeError):
            genre = None
        
        try:
            image_url = r[4]
        except IndexError:
            image_url = None
        
        ratings_dict.append({
            "Title": title,
            "Rating": rating,
            "Author": author,
            "Genre": genre,
            "ImageURL": image_url
        })

    
    return render_template("shelve.html", navshelve=True, ratings=ratings_dict)
@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if not session.get('user_id'):
        flash("Please login to access recommendations", "danger")
        return redirect(url_for('login'))  # If not logged in, redirect user to login page

    user_id = session['user_id']
    recommendations = []
    search_results = []

    if request.method == 'POST':
        book_title = request.form.get('book_title')
        if book_title:
            search_results, _ = get_recommendation(book_title=book_title)

    if user_id:
        recommendations, _ = get_recommendation(user_id=user_id)

    return render_template("recommend.html", navrecommend=True, recommendations=recommendations, search_results=search_results)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user_id = form.user_id.data
        password = form.password.data
        
        # Verify the static password
        if password == 'admin':
            # Retrieve user_id from the database
            db_user_id = get_reviewer_id(user_id)
            if db_user_id:
                # Login successful, store user_id in session
                session['user_id'] = db_user_id
                flash(f'Login successful for user ID: {db_user_id}', 'success')
                print(f"DEBUG: User logged in with user_id: {session['user_id']}")
                return redirect(url_for('catalog'))
            else:
                # Invalid user_id
                flash('Invalid user ID', 'danger')
        else:
            # Invalid password
            flash('Invalid password', 'danger')
    
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "success")
    print("DEBUG: User logged out.")
    return redirect(url_for('login'))
