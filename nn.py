import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


D_IN = 134
D_OUT = 2
FC_HIDDEN = [D_IN,150,150,100,D_OUT]
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
EPOCHS = 10000
L2_FACTOR = 0.001

class simple_net(torch.nn.Module):
    def __init__(self):
        super(simple_net,self).__init__()
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(FC_HIDDEN[i],FC_HIDDEN[i+1])
                for i in range(len(FC_HIDDEN)-1)
             ])
        self.bns = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(FC_HIDDEN[i])
                for i in range(len(FC_HIDDEN)-1)
             ])
        self.softmax = torch.nn.Softmax()

    def forward(self,input):
        for i in range(len(FC_HIDDEN)-1):
            input = self.bns[i](input)
            if i != 0:
                input = torch.nn.functional.relu(input)
            input = self.linears[i](input)
        input = self.softmax(input)
        return input


class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()


def l2norm(parameters):
    r = 0
    for W in parameters:
        if r is None:
            r = torch.norm(W,2)
        else:
            r += torch.norm(W,2)
    return L2_FACTOR*r

def main():
    data_x = pd.read_csv('dataset/gradcafe/cs_preprocessed_x.csv', usecols=[
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    university_names = pd.read_csv('dataset/gradcafe/cs_preprocessed_X.csv', usecols=[5]).values
    data_y = pd.read_csv('dataset/gradcafe/cs_preprocessed_y.csv')
    labels = np.array(data_y)
    labels = np.array([1 if labels[i][0] else 0 for i in range(len(labels))])
    # feature_list = data_x.columns
    le = LabelEncoder()
    le.fit(university_names)
    le_encoded = le.transform(university_names).reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(le_encoded)
    ohe_encoded = ohe.transform(le_encoded)
    data_x = np.hstack((data_x, ohe_encoded))
    print(data_x.shape)
    #
    # pca = PCA(n_components=40)
    # principalComponents = pca.fit_transform(data_x)
    # principalDf = pd.DataFrame(data = principalComponents)

    features = np.array(data_x)




    #partition into train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 2333)
    num_of_train = train_features.shape[0]
    model = simple_net()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_data = torch.tensor(test_features.astype(float)).float()
    test_labels = torch.tensor(test_labels).long()
    tensorboard = Tensorboard('tensorboard_logs')

    #start to train
    for epoch in range(EPOCHS):
        #test every 10 epochs
        if epoch % 10 == 0:
            y_pred = model.eval()(test_data)
            loss = loss_fn(y_pred,test_labels) + l2norm(model.parameters())
            predtions = torch.argmax(y_pred,1)
            print("Testing ------ Epoch %d"%(epoch),"accuracy ",torch.mean(torch.eq(predtions,test_labels).float()).tolist())
            tensorboard.log_scalar("test_loss", loss.tolist(), epoch)
            tensorboard.log_scalar("test_accuracy",torch.mean(torch.eq(predtions,test_labels).float()).tolist(),epoch)
        #shuffle data and train for one epoch
        indices = np.arange(num_of_train)
        np.random.shuffle(indices)
        for i in range(0,num_of_train,BATCH_SIZE):
            train_data_sample = torch.tensor(np.take(train_features, indices[i:min(num_of_train-1,i+BATCH_SIZE)],axis=0).astype(float)).float()
            train_labels_sample = torch.tensor(np.take(train_labels, indices[i:min(num_of_train-1,i+BATCH_SIZE)],axis=0).astype(float)).long()
            y_pred = model(train_data_sample)

            # Compute and print loss.
            loss = loss_fn(y_pred, train_labels_sample) + l2norm(model.parameters())
            if epoch % 10 == 0 and i == 0:
                predtions = torch.argmax(y_pred,1)
                print("Traning EPOCH %d loss: %f"%(epoch,loss),"accuracy ",torch.mean(torch.eq(predtions,train_labels_sample).float()).tolist())
                tensorboard.log_scalar("train_loss", loss.tolist(), epoch)
                tensorboard.log_scalar("train_accuracy",torch.mean(torch.eq(predtions,train_labels_sample).float()).tolist(),epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
