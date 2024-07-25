# from flask import Flask,request
# from flask_cors import CORS, cross_origin
# from flask_restful import Resource,Api
# from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity
# from flask_jwt_extended import JWTManager
# from flask_sqlalchemy import SQLAlchemy
# from uuid import uuid4
# from sqlalchemy import func
  
# db = SQLAlchemy()
  

# app=Flask(__name__)
# app.config['SECRET_KEY']='Sudiksha'
# CORS(app, origins=["*"] )
# app.config["SECRET_KEY"] = "SUDIKSHYA"
# app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@localhost/Project"


# # SQLALCHEMY_TRACK_NOTIFICATIONS=False
# # SQLALCHEMY_ECHO=True
# jwt=JWTManager(app)
# api=Api(app)
# db.init_app(app)

  
# class User(db.Model):
#      __tablename__ = "loginusers"
#      id = db.Column(db.String(11), primary_key=True, unique=True, default=get_uuid)
#      email = db.Column(db.String(150), unique=True)
#      password = db.Column(db.Text, nullable=False)
#      with app.app_context():
#       db.create_all() 
# class Registration(Resource):
#           def post(self):
#             data = request.get_json()
#             email = data.get('email')
#             password = data.get('password')

#             if not email or not password:
#              return {'message': "Missing email or password"}, 400

#         # Case-insensitive query
#           existing_user = User.query.filter(func.lower(User.email) == func.lower(email)).first()

#           if existing_user:
#            return {'message': 'Email already taken'}, 400

#           new_user = User(email=email, password=password)
#           db.session.add(new_user)
#           db.session.commit()

#         return {'message': "Registered successfully"}, 200

                  
# class Login(Resource):
#         def post(self):
#             data=request.get_json()
#             email=data['email']
#             password=data['password']
#             user=User.query.filter_by(email=email).first()
#             if user and user.password==password:
#                 access_token=create_access_token(identity=user.id)
#                 return {
#                     'access_token':access_token
#                 },200
#             return {
#                   'message':"invalid credentials "
#                        },401
        
# @app.route('/')
# def hello():
#     return "hello everyone"
# api.add_resource(Registration,'/register')
# api.add_resource(Login,'/login')


# if __name__=="__main__":
#     app.run(debug=True)