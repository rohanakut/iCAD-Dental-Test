import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
import time
import copy
from sklearn.metrics import confusion_matrix

##Custom Dataloader
class CustomDataset(Dataset):
	def __init__(self):
		self.imgs_path = "../Imagefiles/"
		file_list = glob.glob(self.imgs_path + "*")
		self.data = []
		for class_path in file_list:
			class_name = class_path.split("/")[-1]
			for img_path in glob.glob(class_path + "/*.png"):
				self.data.append([img_path, class_name])
		self.class_map = {"lower_images" : 0, "upper_images": 1}
		self.img_dim = (416, 416)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		##preprocessing
		img_path, class_name = self.data[idx]
		img = cv2.imread(img_path)
		##resizing
		img = cv2.resize(img, self.img_dim)
		class_id = self.class_map[class_name]
		class_id = torch.tensor([class_id])
		#histogram qequalisation
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		lab_planes  = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(2,2))
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		bgr_changed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		#normalisation
		norm = cv2.normalize(bgr_changed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		img_tensor = torch.from_numpy(norm)
		img_tensor = img_tensor.permute(2, 0, 1)
		return img_tensor, class_id

#download pretrained model
def model():
	model_conv = models.densenet121(pretrained=True)
	for param in model_conv.parameters():
		param.requires_grad = False
	num_ftrs = model_conv.classifier.in_features
	model_conv.classifier = nn.Linear(num_ftrs, 2)

	model_conv = model_conv.to(device)

	return model_conv

def train_model(model, criterion, optimizer, scheduler, train_size,val_size,num_epochs=1):
	since = time.time()
	best_acc = 0.0
	example_ct =0
	batch_ct =0
	min_val_loss = np.Inf
	early_stop = False
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		model.train()
		running_loss = 0.0
		running_corrects = 0
		##training loop
		for inputs, labels in tqdm(train_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			inputs = inputs.float()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				labels = labels.reshape(len(labels))
				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			example_ct +=  len(inputs)
			batch_ct += 1
			# statistics
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			scheduler.step()
		epoch_loss_train = running_loss / train_size
		epoch_acc_train = running_corrects.double() / train_size
		print('{} Loss: {:.4f} Acc: {:.4f}'.format(
			'Train', epoch_loss_train, epoch_acc_train))

		##validation loop
		for inputs, labels in tqdm(val_loader):
			model.eval()
			running_loss = 0.0
			running_corrects = 0
			inputs = inputs.to(device)
			labels = labels.to(device) 
			example_ct +=  len(inputs)
			batch_ct += 1
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
		epoch_loss_val = running_loss / val_size
		epoch_acc_val = running_corrects.double() / val_size
		if(epoch_acc_val>best_acc):
			best_acc = epoch_acc_val
			best_model_wts = copy.deepcopy(model.state_dict())
		print('{} Loss: {:.4f} Acc: {:.4f}'.format(
			'Validation', epoch_loss_val, epoch_acc_val))
		if epoch_loss_val < min_val_loss:
			epochs_no_improve = 0
			min_val_loss = epoch_loss_val
		else:
			epochs_no_improve += 1
		if epochs_no_improve == 6:
			early_stop = True
		if early_stop:
			torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': epoch_loss_val,
			}, './early_stopped.pth')
			print("Stopped")
			break


	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	#save model once training is completed
	torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': epoch_loss_val,
			}, './save.pth')
	return model

def test_model(test_loader,test_size):
	corrects = 0
	all_preds = []
	y = []
	for inputs, labels in tqdm(test_loader):
				inputs = inputs.to(device)
				labels = labels.to(device)
				inputs = inputs.float()
				labels = labels.reshape(len(labels))
				outputs = model_conv(inputs)
				_, preds = torch.max(outputs, 1)
				corrects += torch.sum(preds == labels.data)
				preds = preds.to("cpu")
				labels = labels.to("cpu")
				preds = preds.numpy()
				labels = labels.numpy()
				all_preds = np.append(all_preds,preds)
				y = np.append(y,labels)
	epoch_acc = corrects.float() / test_size
	tn, fp, fn, tp = confusion_matrix(y, all_preds).ravel()
	sensitivity  =tp / (tn+fp)
	specificity = tn/(tn+fp)
	epoch_acc = epoch_acc.to("cpu")
	epoch_acc = epoch_acc.numpy()
	print("Test Accuracy is:",epoch_acc*100)
	print("Sensitivity is : ", sensitivity*100)
	print("Specificity is :", specificity*100)
	print(confusion_matrix(y, all_preds))


if __name__ == "__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataset = CustomDataset()
	batch_size = 4
	validation_split = .2
	shuffle_dataset = True
	random_seed= 42
	epochs = 10

	# Creating data indices for training and validation splits:
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))
	if shuffle_dataset :
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, test_indices = indices[split:], indices[:split]

	train_size = len(train_indices)
	indices = list(range(train_size))
	split = int(np.floor(0.1 * dataset_size))
	if shuffle_dataset :
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	print(len(train_indices))
	print(len(val_indices))
	print(len(test_indices))

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	test_sampler = SubsetRandomSampler(test_indices)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
											sampler=train_sampler)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=test_sampler)
	val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=valid_sampler)
		
	data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
	model_conv = model()
	criterion = nn.CrossEntropyLoss()
	optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.01, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
	model_conv = train_model(model_conv, criterion, optimizer_conv,
						 exp_lr_scheduler,len(train_indices),len(val_indices), num_epochs=epochs)
	test_model(test_loader,len(test_indices))