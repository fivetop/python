import pickle
import os
import multiprocessing
import math

from django.utils import timezone
from django.db import models
from pybrain.structure import GaussianLayer, LSTMLayer, SigmoidLayer, SoftmaxLayer, LinearLayer, TanhLayer
from random import random
from pybrain.structure import FeedForwardNetwork, RecurrentNetwork, FullConnection
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from coursework import settings


class NN(models.Model):
    dsTypes = (('SV', 'supervised'), ('CL', 'classification'), )
    name = models.CharField(max_length=50)
    date = models.DateTimeField('run date')
    inputs = models.FileField(upload_to='documents')
    email = models.EmailField()
    typeDS = models.CharField(max_length=2, choices=dsTypes, default='SV')
    populationSize = models.PositiveIntegerField()
    numberOfNeurons = models.PositiveIntegerField()
    epochCount = models.PositiveIntegerField()

    def __unicode__(self):
        return self.name

    def getCompletion(self):
        arr = list(GA.objects.filter(net=self.id))
        summ = 0
        for ga in arr:
            summ += ga.getCompletion()
        return int((summ / (len(arr) * self.populationSize * self.epochCount)) * 100)

    def anotherGeneration(self):
        self.start()

    def handle_uploaded_file(self, f):
        destination = open(self.inputs.__str__(), 'wb+')
        for chunk in f.chunks():
            destination.write(chunk)
        destination.close()

    # Works with data sets
    def makeSupervisedDS(self, inp, target, inp_samples, target_samples):
        ds = SupervisedDataSet(inp, target)
        i = 0
        for _ in inp_samples:
            ds.addSample(tuple(inp_samples[i]), tuple(target_samples[i]))
            i += 1
        return ds

    def makeClassificationDS(self, inp, inp_samples, target_samples):
        ds = ClassificationDataSet(inp)
        i = 0
        for _ in inp_samples:
            ds.addSample(inp_samples[i], target_samples[i])
            i += 1
        ds._convertToOneOfMany()
        return ds

    # Working with files
    def parseLine(self, line):
        res = []
        work = line.split(' ')
        for i in range(len(work)):
            res.append(int(work[i]))
        return tuple(res)

    def readFile(self):
        self.inputs.open(mode='rb')
        line = self.inputs.readline()
        inpt = []
        target = []
        if line == 'INPUT\n':
            self.sequential = False
            line = self.inputs.readline()
            counter = 0
            while line != 'TARGET\n':
                inpt.append(self.parseLine(line))
                line = self.inputs.readline()
                counter += 1
            line = self.inputs.readline()
            self.input_length = counter
            while counter != 0:
                target.append(self.parseLine(line))
                line = self.inputs.readline()
                counter -= 1
            print (inpt)
            print (target)
        else:
            self.sequential = True
            inpt.append(self.parseLine(line))
            for line in self.inputs:
                inpt.append(self.parseLine(line))
            print (inpt)
        self.inputs.close()
        return inpt, target

    def getURL(self):
        return 'http://127.0.0.1:8000/ai/' + str(self.id)

    # noinspection PyBroadException
    def start(self):
        inp, tar = self.readFile()
        try:
            os.mkdir(settings.MEDIA_ROOT + 'xmls/' + str(self.id))
        except:
            pass
        try:
            os.mkdir(settings.MEDIA_ROOT + 'matrix/' + str(self.id))
        except:
            pass
        if self.typeDS == 'SV':
            ds = self.makeSupervisedDS(len(inp[0]), len(tar[0]), inp, tar)
        else:
            ds = self.makeClassificationDS(len(inp[0]), inp, tar)
        nameGA = len(GA.objects.filter(net=self)) + 1
        ga = GA(net=self, name=nameGA, start=timezone.now(), networks=(self.epochCount * self.populationSize))
        ga.save()
        p = multiprocessing.Process(target=ga.run,
                                    args=(ds, self.epochCount, self.numberOfNeurons, self.populationSize,))
        p.start()


class GA(models.Model):
    net = models.ForeignKey(NN)
    name = models.CharField(max_length=50)
    start = models.DateTimeField()
    populations = models.PositiveIntegerField(null=True, blank=True)
    networks = models.PositiveIntegerField()

    func = [GaussianLayer, SigmoidLayer, SoftmaxLayer, LinearLayer, TanhLayer]

    def getCompletion(self):
        a = self.networks
        b = len(list(Network.objects.filter(net=self.id)))
        res = 100. * b / a
        return res

    def getURL(self):
        return 'http://127.0.0.1:8000/ai/' + str(self.net.id) + '/' + str(self.name)

    def fillConnections(self, net, addedStack, stackToGo, layers):
        connections = []
        recurrentConnections = []
        if len(stackToGo) == 0:
            return connections, recurrentConnections
        ways = []
        futureStack = []
        for neuron in stackToGo:
            way = []
            for w in ways:
                if w.count(neuron) != 0:
                    way = w
                else:
                    continue
            if not way:
                way.append(neuron)
                ways.append(way)
            for connection in range(len(net[neuron])):
                if net[neuron][connection] == 1:
                    if addedStack.count(connection) != 0:
                        recurrentConnections.append(FullConnection(layers[neuron], layers[connection]))
                    else:
                        if stackToGo.count(connection) != 0:
                            if way.count(connection) != 0:
                                recurrentConnections.append(FullConnection(layers[neuron], layers[connection]))
                            else:
                                flag = True
                                for w in ways:
                                    if w.count(connection) != 0:
                                        connections.append(FullConnection(layers[neuron], layers[connection]))
                                        for n in w:
                                            way.append(n)
                                        ways.pop(ways.index(w))
                                        flag = False
                                        break
                                    else:
                                        continue
                                if flag:
                                    way.append(connection)
                                    connections.append(FullConnection(layers[neuron], layers[connection]))
                        else:
                            connections.append(FullConnection(layers[neuron], layers[connection]))
                            futureStack.append(connection)
                else:
                    continue
        for v in stackToGo:
            addedStack.append(v)
        c, rc = self.fillConnections(net, addedStack, futureStack, layers)
        for con in c:
            connections.append(con)
        for rcon in rc:
            recurrentConnections.append(rcon)
        return connections, recurrentConnections

    def buildNN(self, net, functions, inp, out):
        layers = []

        inLayer = self.func[functions[0]](inp)
        layers.append(inLayer)
        outLayer = self.func[functions[-1]](out)

        for neural in range(1, len(net) - 1):
            layers.append(self.func[functions[neural]](1))
        layers.append(outLayer)

        connections, recConnections = self.fillConnections(net, [], [0], layers)
        if len(recConnections) == 0:
            n = FeedForwardNetwork()
        else:
            n = RecurrentNetwork()
        n.addInputModule(inLayer)
        for layer in range(1, len(layers) - 1):
            n.addModule(layers[layer])
        n.addOutputModule(outLayer)

        for con in connections:
            n.addConnection(con)
        for rcon in recConnections:
            n.addRecurrentConnection(rcon)
        n.sortModules()
        return n

    def mutateFunc(self, functions, rate):
        result = []
        for f in range(len(functions)):
            if random() < rate:
                result.append(int(round(random() * (len(self.func) - 0.5))))
            else:
                result.append(functions[f])
        return result

    def mutateMatrix(self, matrix, rate):
        result = []
        for i in range(len(matrix)):
            tmp = []
            for j in range(len(matrix[i])):
                if random() < rate:
                    tmp.append(int(round(random())))
                else:
                    tmp.append(matrix[i][j])
            result.append(tmp)
        if result[0].count(1) == 0:
            result = self.mutateMatrix(matrix, rate)
        return result

    def initChild(self, length):
        res = []
        for i in range(length):
            tmp = []
            for j in range(length):
                tmp.append(0)
            res.append(tmp)
        return res

    def crossNet(self, motherMatrix, motherFunc, fatherMatrix, fatherFunc):
        length = len(motherMatrix[0]) ** 2
        motherPart = round(length * random())
        probability = motherPart / length
        child = self.initChild(len(motherMatrix))
        childFunc = []
        for i in range(len(fatherMatrix)):
            mother = random() < probability
            if mother:
                childFunc.append(motherFunc[i])
            else:
                childFunc.append(fatherFunc[i])
            for j in range(len(motherMatrix[i])):
                if mother:
                    child[i][j] = motherMatrix[i][j]
                else:
                    child[i][j] = fatherMatrix[i][j]
        if child[0].count(1) == 0:
            pass
        return [child, childFunc]

    def init(self, popSize, matrixSize):
        population = []
        for i in range(popSize):
            matrix = []
            function = []
            for neuron in range(matrixSize):
                tmp = []
                function.append(int(round(random() * (len(self.func) - 0.5))))
                for con in range(matrixSize):
                    if random() < 0.5:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                matrix.append(tmp)
            population.append([matrix, function])
        return population

    # noinspection PyBroadException
    def run(self, trnDS, epochs, neuronsCount, popSize):
        try:
            os.mkdir(settings.MEDIA_ROOT + 'xmls/' + str(self.net.id) + '/' + str(self.name))
        except:
            pass
        try:
            os.mkdir(settings.MEDIA_ROOT + 'matrix/' + str(self.net.id) + '/' + str(self.name))
        except:
            pass
        population = self.init(popSize, neuronsCount)
        proportion = []
        for i in range(1, popSize + 1):
            proportion.append(i ** 2)
        summ = sum(proportion)
        self.runEvol(epochs, trnDS, popSize, summ, proportion, population, 0)

    def generateNewPop(self, popSize, summ, proportion, population, rang):
        newPopulation = []
        for child in range(popSize):
            rand = int(round(random() * summ))
            already = 0
            for prop in proportion:
                if rand <= already + prop:
                    motherIndex = int(math.sqrt(prop))
                    break
                else:
                    already = already + prop
            rand = int(round(random() * summ))
            already = 0
            for prop in proportion:
                if rand <= already + prop:
                    fatherIndex = int(math.sqrt(prop))
                    break
                else:
                    already = already + prop
            motherMatrix = population[rang[-motherIndex]][0]
            motherFunc = population[rang[-motherIndex]][1]
            fatherMatrix = population[rang[-fatherIndex]][0]
            fatherFunc = population[rang[-fatherIndex]][1]
            childToAdd = self.crossNet(motherMatrix, motherFunc, fatherMatrix, fatherFunc)
            if random() < 0.5:
                childToAdd[0] = self.mutateMatrix(childToAdd[0], 0.4)
                childToAdd[1] = self.mutateFunc(childToAdd[1], 0.4)
            newPopulation.append(childToAdd)
        return newPopulation

    def runEvol(self, epochs, trnDS, popSize, summ, proportion, population, currentEpoch):
        if currentEpoch != epochs:
            net = []
            survivalRate = []
            for p in population:
                #TODO multyproc
                net.append(self.buildNN(p[0], p[1], len(trnDS['input'][0]), len(trnDS['target'][0])))
                netName = len(Network.objects.filter(net=self.id)) + 1
                network = Network(net=self, name=netName, population=currentEpoch)
                network.save()
                res = network.run(net[-1], trnDS, [p[1], p[0]])
                survivalRate.append(res)
            rang = []
            for sr in range(len(survivalRate)):
                rang.append(survivalRate.index(min(survivalRate)))
                survivalRate[rang[-1]] = 101
            population = self.generateNewPop(popSize, summ, proportion, population, rang)
            self.runEvol(epochs, trnDS, popSize, summ, proportion, population, currentEpoch + 1)
        else:
            pass


class Network(models.Model):
    net = models.ForeignKey(GA)
    name = models.CharField(max_length=50)
    population = models.PositiveIntegerField()
    best_run = models.FloatField(null=True, blank=True)

    func = ['GaussianLayer', 'SigmoidLayer', 'SoftmaxLayer', 'LinearLayer', 'TanhLayer']

    # noinspection PyBroadException
    def train(self, net, ds):
        try:
            trainer = BackpropTrainer(net, ds)
            out = trainer.trainUntilConvergence(maxEpochs=350)
        except:
            out = [[101.], [101.]]
            pass
        return (min(out[0]) + min(out[1])) / 2

    def getMatrixURL(self):
        return 'http://127.0.0.1:8000/media/matrix/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(
            self.name)

    def getXmlURL(self):
        return 'http://127.0.0.1:8000/media/xmls/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(
            self.name)

    def getURL(self):
        return 'http://127.0.0.1:8000/ai/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(self.name)

    def getMatrix(self):
        res = []
        f = open(
            settings.MEDIA_ROOT + 'matrix/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(self.name),
            mode='rb')
        line = f.readline()
        functions = line.split(', ')
        r = []
        for l in functions:
            r.append(int(l.replace('\n', '')))
        res.append(r)
        line = f.readline()
        matrix = []
        while True:
            if not line:
                break
            tmp = line.split(', ')
            r = []
            for t in tmp:
                r.append(int(t.replace('\n', '')))
            matrix.append(r)
            line = f.readline()
        f.close()
        res.append(matrix)
        return res

    def getMat(self):
        return self.getMatrix()[1]

    def getMatSize(self):
        return len(self.getMat)

    def getFunc(self):
        return self.getMatrix()[0]

    def getFuncText(self):
        res = ''
        functions = self.getFunc()
        i = 0
        for f in functions:
            res += '' + str(i) + ': ' + self.func[f] + ', '
            i += 1
        return res

    # noinspection PyBroadException
    def saveMatrix(self, data):
        try:
            os.mkdir(settings.MEDIA_ROOT + 'matrix/' + str(self.net.net.id) + '/' + str(self.net.name))
        except:
            pass
        f = open(
            settings.MEDIA_ROOT + 'matrix/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(self.name),
            'w')
        write = ''
        for d in range(len(data[0]) - 1):
            write += str(data[0][d]) + ', '
        write += str(data[0][len(data[0]) - 1])

        write += os.linesep

        for i in range(len(data[1])):
            for d in range(len(data[1][i]) - 1):
                write += str(data[1][i][d]) + ', '
            write += str(data[1][i][len(data[1][i]) - 1])
            write += os.linesep

        f.write(write)
        f.close()

    # noinspection PyBroadException
    def run(self, n, ds, data):
        try:
            os.mkdir(settings.MEDIA_ROOT + 'xmls/' + str(self.net.net.id) + '/' + str(self.net.name))
        except:
            pass
        p = multiprocessing.Process(target=self.saveMatrix,
                                    args=(data,))
        p.start()
        self.best_run = self.train(n, ds.copy())
        xml = open(
            settings.MEDIA_ROOT + 'xmls/' + str(self.net.net.id) + '/' + str(self.net.name) + '/' + str(self.name), 'w')
        pickle.dump(n, xml)
        xml.close()

        self.save()
        return self.best_run
