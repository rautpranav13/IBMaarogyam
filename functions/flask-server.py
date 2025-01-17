import os
import sys
from flask_app.app import app as flask_app

def handler(event, context):
    from flask_lambda import handler as aws_lambda_handler
    return aws_lambda_handler(event, context, flask_app)
