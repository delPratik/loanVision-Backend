from flask_sqlalchemy import SQLAlchemy
from uuid import uuid4

db = SQLAlchemy()

def get_uuid():
    return uuid4().hex

class AdminUser(db.Model):  # Updated class name
    __tablename__ = "admin_user"  # Updated table name

    id = db.Column(db.String(32), primary_key=True, default=get_uuid)
    useremail = db.Column(db.String(150), unique=True, nullable=False)
    userpassword = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<AdminUser {self.useremail}>'

class Register(db.Model):
    __tablename__ = "register"

    id = db.Column(db.String(32), primary_key=True, default=get_uuid)
    username = db.Column(db.String(150), nullable=False)
    useremail = db.Column(db.String(150), unique=True, nullable=False)
    userpassword = db.Column(db.Text, nullable=False)
    userphone = db.Column(db.Text, nullable=False)
    confirmpassword = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Register {self.username} - {self.useremail}>'
