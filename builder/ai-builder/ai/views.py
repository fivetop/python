# Create your views here.
import multiprocessing

from django.utils import timezone
from django.shortcuts import render, get_object_or_404, render_to_response
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.template import Context, RequestContext
from django import forms

from ai.forms import UploadFileForm
from ai.models import NN, GA, Network


def index(request):
    #context = Context()
    nw = NN.objects.all()
    return render(request, 'ai/index.html', {'nw': nw})


def details(request, nw_id):
    if request.method == 'POST':
        nn = get_object_or_404(NN, pk=nw_id)
        nn.anotherGeneration()
    nw = get_object_or_404(NN, pk=nw_id)
    gas = GA.objects.filter(net=nw_id)
    return render_to_response(
        'ai/details.html',
        {'form': forms.BaseForm, 'network': nw, 'gas': gas},
        context_instance=RequestContext(request)
    )
    #return render(request, 'ai/details.html', {'network': nw, 'gas': gas, 'form': forms.BaseForm})


def detailsGA(request, nw_id, ga_id):
    nn = get_object_or_404(NN, pk=nw_id)
    ga = GA.objects.filter(net=nn.id).get(name=ga_id)
    networks = list(Network.objects.filter(net=ga.id))
    return render(request, 'ai/details_ga.html', {'ga': ga, 'nn': nn, 'networks': networks})


def detailNetwork(request, nw_id, ga_id, net_id):
    nn = get_object_or_404(NN, pk=nw_id)
    ga = GA.objects.filter(net=nn.id).get(name=ga_id)
    network = Network.objects.filter(net=ga.id).get(name=net_id)
    return render(request, 'ai/detail_net.html', {'ga': ga, 'nn': nn, 'network': network})


def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            network = NN(name=request.POST['title'],
                         date=timezone.now(),
                         inputs=request.FILES['file'],
                         typeDS=request.POST['typeDS'],
                         populationSize=int(request.POST['populationSize']),
                         numberOfNeurons=int(request.POST['numberOfNeurons']),
                         epochCount=int(request.POST['epochCount']))
            network.save()
            p = multiprocessing.Process(target=network.start, args=())
            p.start()
            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('ai.views.details', args=(network.id,)))
    else:
        form = UploadFileForm() # A empty, unbound form
        # Render list page with the documents and the form
    return render_to_response(
        'ai/upload.html',
        {'form': form},
        context_instance=RequestContext(request)
    )