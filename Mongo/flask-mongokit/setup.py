"""
Flask-MongoKit
--------------

Flask-MongoKit simplifies to use MongoKit, a powerful MongoDB ORM in Flask
applications.

Links
`````

* `documentation <http://packages.python.org/Flask-MongoKit>`_
* `development version <http://github.com/jarus/flask-mongokit/zipball/master#egg=Flask-MongoKit-dev>`_
* `MongoKit <http://namlook.github.com/mongokit/>`_
* `Flask <http://flask.pocoo.org>`_

"""
import sys
from setuptools import setup

install_requires = [
    "Flask",
    "MongoKit"
]

if sys.version_info < (2, 6):
    install_requires.append('simplejson')

setup(
    name='Flask-MongoKit',
    version='0.6',
    url='http://github.com/jarus/flask-mongokit',
    license='BSD',
    author='Christoph Heer',
    author_email='Christoph.Heer@googlemail.com',
    description='A Flask extension simplifies to use MongoKit',
    long_description=__doc__,
    py_modules=['flask_mongokit'],
    zip_safe=False,
    platforms='any',
    install_requires=install_requires,
    test_suite='tests.suite',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
