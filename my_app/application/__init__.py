from flask import Flask
from config import Config
from logging.config import dictConfig
from flask_sqlalchemy import SQLAlchemy


# Configure logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})



app = Flask(__name__)


app.config.from_object(Config)
# db = SQLAlchemy(app)

from application import routes

# with app.app_context():
#     db.create_all()



if __name__ == "__main__":
    app.run(debug=True)