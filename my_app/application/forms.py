from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length

class LoginForm(FlaskForm):
    user_id = StringField("User ID", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=5, max=15)])
    submit = SubmitField("Login")
