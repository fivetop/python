from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField

class classes():
    reset = SubmitField('Reset')


class CreateTask(FlaskForm):
    title = StringField('Task Title')
    shortdesc = StringField('Short Description')
    priority = IntegerField('Priority')
    create = SubmitField('Create')

class DeleteTask(FlaskForm):
    key = StringField('Task ID')
    title = StringField('Task Title')
    delete = SubmitField('Delete')

class UpdateTask(FlaskForm):
    key = StringField('Task Key')
    shortdesc = StringField('Short Description')
    update = SubmitField('Update')

class ResetTask(FlaskForm):
    reset = SubmitField('Reset')
