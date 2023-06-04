from django.shortcuts import render
from .forms import LoanForm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

def LoanPredictionView(request):
    if request.method == 'POST':
        form = LoanForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            features = np.array([[form_data['home_ownership_cat'], form_data['income_cat'], form_data['term_cat'], form_data['application_type_cat'], form_data['purpose_cat'], form_data['interest_payment_cat'], form_data['loan_condition_cat'], form_data['grade_cat'], form_data['emp_length_int'], form_data['annual_inc'], form_data['loan_amount'], form_data['interest_rate'], form_data['dti'], form_data['total_pymnt'], form_data['total_rec_prncp'], form_data['recoveries'], form_data['installment'], form_data['month_diff']]], dtype=np.float32)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            model = tf.saved_model.load('Credit scoring 1')
            predictions = model(features_scaled)
            loan_condition = np.argmax(predictions)
            return render(request, 'prediction.html', {'loan_condition': loan_condition})
    else:
        form = LoanForm()
    return render(request, 'index.html', {'form': form})
