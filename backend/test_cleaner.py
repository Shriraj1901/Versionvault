from cleaner import clean_text

messy = "This  is   auth-\ncation text.\n\n\n\n\nWith   weird spacing.   \n\n"
print(repr(clean_text(messy)))