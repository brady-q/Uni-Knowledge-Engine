"""Data access. Every SQL statement in the application is written in here.

Repositories take an ``AsyncSession``, speak in ORM models, and never commit:
the transaction boundary belongs to the service that called them, so one
service call stays one transaction even when it touches several repositories.
"""
