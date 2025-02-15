import os
import django

# 1. Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# 2. Initialize Django
django.setup()

# 3. Import the model
from service.models import Product


# product = Product(name="Laptop", price=1100.00, description="A power laptop")
# product.save()
# product = Product(name="Laptop", price=1300.00, description="A amazing laptop")
# product.save()


# 4. Fetch all products
# products = Product.objects.all()

# # 5. Print the results
# for product in products:
#     print(product.name, product.price, product.description)

# product = Product.objects.get(id=1)
# for field in product._meta.get_fields():
#     print(field.name)

products = Product.objects.filter(name="Laptop")
for row in products: 
    print(row.name)

print(Product._meta.get_fields())

