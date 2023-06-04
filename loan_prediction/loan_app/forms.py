from django import forms
from .models import Loan
from sklearn.preprocessing import StandardScaler

class LoanForm(forms.ModelForm):
    class Meta:
        model = Loan
        fields = '__all__'

    def clean(self):
        cleaned_data = super().clean()
        features = ['emp_length_int', 'annual_inc', 'loan_amount', 'interest_rate', 'dti', 'total_pymnt', 'total_rec_prncp', 'recoveries', 'installment', 'month_diff']
        scaler = StandardScaler()
        for feature in features:
            cleaned_data[feature] = scaler.fit_transform([[cleaned_data[feature]]])[0]
        return cleaned_data
