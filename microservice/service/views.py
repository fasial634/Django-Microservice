from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .model_loader import predict_sentiment

class PredictView(APIView):
    def post(self, request):
   
        try:
            probability, label = predict_sentiment(request.data["text"])
            print(request.data["text"])
            return Response(
                {"probability": probability, "label": label},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

