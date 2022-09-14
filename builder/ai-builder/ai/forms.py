from django import forms
# Create your models here.


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField(label='Select a file')
    typeDS = forms.ChoiceField(widget=forms.Select(),
                               choices=(('SV', 'supervised'), ('CL', 'classification'),),
                               initial='SV',
                               required=True,
                               label='Dataset Type')
    populationSize = forms.IntegerField(label='Population Size')
    numberOfNeurons = forms.IntegerField(label='Number of neurons')
    epochCount = forms.IntegerField(label='Number of epochs')