from django.db import models

class Product(models.Model): 
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # def apply_discount(self, discount_percentage):
    #     self.price = self.price * (1 - discount_percentage / 100)
    #     self.save()

    # def __str__(self): 
    #     return self.name