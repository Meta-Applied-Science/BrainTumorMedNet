"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y,_) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              log:str ='log.pth') -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        # X,y and image path for XAI
        for batch, (X, y,paths) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            mismatched = test_pred_labels != y  # Boolean mask
            # wrong_indices = torch.nonzero(mismatched).squeeze().view(-1)
            wrong_indices = torch.nonzero(mismatched, as_tuple=False).view(-1)  # ép 1D

            # print(wrong_indices)
            # print(wrong_indices.numel() )

            if wrong_indices.numel() > 0:
                with open("/home/ma012/AlexServer/log/path_predict.txt", "a", encoding="utf-8") as file:
                    for i in range(len(wrong_indices)):
                        file.write(f"{paths[wrong_indices[i]]}\n")


    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write(f"\n test loss: {test_loss} | test accuracy: {test_acc}\n===================================================================")

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

def trainVal(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,  # Renamed from test_dataloader
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          save_path: str = "model.pth",
          log:str ='log.txt') -> Dict[str, List]:
    """Trains and validates a PyTorch model.

    Args:
        model: A PyTorch model to be trained and validated.
        train_dataloader: A DataLoader instance for training.
        val_dataloader: A DataLoader instance for validation.
        optimizer: A PyTorch optimizer to minimize the loss function.
        loss_fn: A PyTorch loss function.
        epochs: Number of epochs to train for.
        device: Target device (e.g., "cuda" or "cpu").

    Returns:
        A dictionary containing training and validation loss/accuracy metrics.
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],  # Changed from test_loss
               "val_acc": []}   # Changed from test_acc
    
    # Move model to target device
    model.to(device)

    # Loop through epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        val_loss, val_acc = test_step(model=model,  # Using test_step for validation
                                      dataloader=val_dataloader,  # Changed from test_dataloader
                                      loss_fn=loss_fn,
                                      device=device)

        # Print training and validation metrics
        print(
          f"Epoch: {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "  # Changed from test_loss
          f"Val Acc: {val_acc:.4f}"       # Changed from test_acc
        )

        with open(f"{log}","a",encoding = "utf-8") as file:
            file.write(
          f"\n Epoch: {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "  # Changed from test_loss
          f"Val Acc: {val_acc:.4f}"       # Changed from test_acc
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)  # Changed from test_loss
        results["val_acc"].append(val_acc)    # Changed from test_acc
    
    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write("\n=======================================================================================")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    return results


def load_model_checkpoint(model: torch.nn.Module, 
                          checkpoint_path: str, 
                          device: torch.device) -> torch.nn.Module:
    # state_dict = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(state_dict)
    # model.to(device)
    # model.eval()
    # return model
    #### GPT FIX
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Remap keys cho classifier head nếu cần
    new_state_dict = {}
    for k, v in state_dict.items():
        # Nếu key bắt đầu bằng "heads." mà không bắt đầu với "heads.head.",
        # chuyển đổi thành "heads.head." + phần còn lại của key.
        if k.startswith("heads.") and not k.startswith("heads.head."):
            new_key = "heads.head." + k[len("heads."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load state dict đã được remap vào model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model

def calculate_sensitivity_specificity(y_true, y_pred, num_classes):

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    sensitivity_per_class = []
    specificity_per_class = []
    
    for i in range(num_classes):
        TP = cm[i, i]  
        FN = np.sum(cm[i, :]) - TP  
        FP = np.sum(cm[:, i]) - TP  
        TN = np.sum(cm) - (TP + FN + FP) 
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        sensitivity_per_class.append(sensitivity)
        specificity_per_class.append(specificity)
    
    sensitivity_per_class = np.array(sensitivity_per_class)
    specificity_per_class = np.array(specificity_per_class)

    avg_sensitivity = np.mean(sensitivity_per_class)
    avg_specificity = np.mean(specificity_per_class)

    return sensitivity_per_class, specificity_per_class, avg_sensitivity, avg_specificity

def old_ensemble_test_step(models: List[torch.nn.Module], 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module,
                       device: torch.device,
                       log: str='log.txt') -> Tuple[float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for model in models:
        model.eval()

    with torch.inference_mode():
        # X,y and path image
        for (X, y,paths) in dataloader:
            X, y = X.to(device), y.to(device)
            ensemble_softmax = None

            for model in models:
                logits = model(X)
                softmax_out = torch.softmax(logits, dim=1)
                if ensemble_softmax is None:
                    ensemble_softmax = softmax_out
                else:
                    ensemble_softmax += softmax_out
 
            ensemble_softmax /= len(models) #(16,3)

            # ignore log(0) using 1e-8 - GPT fix
            # loss = -torch.log(ensemble_softmax.gather(1, y.unsqueeze(1)) + 1e-8).mean()
            # total_loss += loss.item() * X.size(0)loss_fn = torch.nn.CrossEntropyLoss()

            loss = loss_fn(ensemble_softmax, y) 
            total_loss += loss.item()

            preds = ensemble_softmax.argmax(dim=1)
            # print(f"preds:{preds} | shape:{preds.shape}")

            mismatched = preds != y  # Boolean mask
            if mismatched.any():  # Check if there's at least one mismatch
                with open("/home/ma012/AlexServer/log/path_predict.txt", "a", encoding="utf-8") as file:
                    for path, mismatch in zip(paths, mismatched):
                        if mismatch:
                            file.write(f"{path}\n")


            total_correct += (preds == y).sum().item()

            total_samples += X.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write(f"\n test loss: {avg_loss} | test accuracy: {avg_acc}\n===================================================================")


    _, _, avg_sensitivity, avg_specificity = calculate_sensitivity_specificity(all_labels, all_preds, num_classes=3)


    return avg_loss, avg_acc, avg_sensitivity, avg_specificity


def ensemble_test_step(models: List[torch.nn.Module], 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module,
                       device: torch.device,
                       log: str='log.txt') -> Tuple[float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Metrics for Sensitivity and Specificity
    TP, TN, FP, FN = 0, 0, 0, 0  

    for model in models:
        model.eval()

    with torch.inference_mode():
        # X,y and path image
        for (X, y,paths) in dataloader:
            X, y = X.to(device), y.to(device)
            ensemble_softmax = None

            for model in models:
                logits = model(X)
                softmax_out = torch.softmax(logits, dim=1)
                if ensemble_softmax is None:
                    ensemble_softmax = softmax_out
                else:
                    ensemble_softmax += softmax_out
 
            ensemble_softmax /= len(models) #(16,3)

            loss = loss_fn(ensemble_softmax, y) 
            total_loss += loss.item()

            preds = ensemble_softmax.argmax(dim=1)
            # print(f"preds:{preds} | shape:{preds.shape}")
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

            mismatched = preds != y  # Boolean mask
            if mismatched.any():  # Check if there's at least one mismatch
                with open("/home/ma012/AlexServer/log/path_predict.txt", "a", encoding="utf-8") as file:
                    for path, mismatch in zip(paths, mismatched):
                        if mismatch:
                            file.write(f"{path}\n")

            # Compute TP, TN, FP, FN
            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write(f"\n test loss: {avg_loss} | test accuracy: {avg_acc}\n===================================================================")

    return avg_loss, avg_acc, sensitivity, specificity

def my_ensemble_test_step(models: List[torch.nn.Module], 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module,
                       device: torch.device,log: str='log.txt') -> Tuple[float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for model in models:
        model.eval()

    with torch.inference_mode():
        # X,y and path image
        for (X, y,paths) in dataloader:
            X, y = X.to(device), y.to(device)

            ensemble_softmax = None

            classifier_origin = []
    
            for model in models:
                logits = model(X)
                softmax_out = torch.softmax(logits, dim=1) # step 1 - softmax_out_i = p_i (each classifier)

                classifier_origin.append(softmax_out) 
                
                if ensemble_softmax is None:
                    ensemble_softmax = softmax_out.clone() # avoid override softmax_out -> affect to first classfier_origin[0]
                else:
                    ensemble_softmax += softmax_out # step 2 - sum the probs of all classifier for each class 
            
            ### === check copy classifier successfully ====
            # print(f"en:{ensemble_softmax}")
           
            # # Initialize W as a tensor with the same shape as softmax_out
            # W = torch.zeros_like(ensemble_softmax)

            # # Sum the individual softmax outputs
            # for softmax_i in classifier_origin:
            #     W += softmax_i

            # print(f"w:{W}")
            ### ==========================================
            
            # ==== Step 3 ================================
            epsilon = 1e-8  # small value to prevent division by zero
            W = [softmax_i / (ensemble_softmax + epsilon) for softmax_i in classifier_origin] # list
            ### === check constraint of step 3 ====
            # W_tensor = torch.stack(W)  # shape: (num_models, batch_size, num_classes)
            # # print(f"W shape: {W_tensor.shape}") # [4,16,3]

            # W_sum = W_tensor.sum(dim=0)
            # print(f"W_sum shape: {W_sum}")  # should be [16, 3]
            ### =================================================
            # ==== End step 3 ============================

            # ==== Weight Voting ====
            W_tensor = torch.stack(W) #4,16,3

            # you get the element-wise maximum across the 4 tensors you stacked.
            _, Weight_indices= W_tensor.max(dim=0) # (16,3)

            # Step 2: Index's One-hot encode 
            num_classes = W_tensor.shape[0] 
            one_hot = F.one_hot(Weight_indices, num_classes=num_classes)  # shape: (H, W, C)

            # Step 3: Count occurrences of each index in each row
            counts = one_hot.sum(dim=1)  # # shape: (H, C)

            # Step 4: Take the index with the highest count per row
            row_major_indices = counts.argmax(dim=1)  # shape: (H,)

            # print("Most Frequently Index", row_major_indices)

            result_row = []
            softmax_origin_tensor = torch.stack(classifier_origin)
            for i in range(len(row_major_indices)):
                # print(f"i:{i} | row major: {row_major_indices[i]}")
                matrix = row_major_indices[i]

                result = softmax_origin_tensor[matrix][i]  # Pick what sub-matrix and probs respect with each sample i
                # print(f"result:{result}")
                result_row.append(result)

            result_tensor = torch.stack(result_row)

            # print("Most frequent indices:", row_major_indices)
            # print("New tensor with corresponding rows: \n", result_tensor)

            loss = loss_fn(result_tensor,y)
            total_loss += loss.item()
            
            # ====== END weight voting method ====
            
            ### == take argmax of Weight ====
            preds = result_tensor.argmax(dim=1)
            print(f"preds:{preds}")

            # # # ==== Step 4 ================================
            # # # Tạo danh sách chứa các mảng Q
            # Q_list = []

            # # # Khởi tạo Q1
            # Q = W[0]

            # # # Thêm Q1 vào danh sách
            # Q_list.append(Q)

            # # # Lặp qua các phần tử tiếp theo của W để tính Q2, Q3, ...
            # for i in range(1, len(W)):
            #     Q = Q + W[i]  # Cộng dồn W[i] vào Q
            #     Q_list.append(Q)

            # # # Chuyển danh sách Q_list thành mảng
            # Q_matrix = np.array(Q_list)

            # # # Hiển thị kết quả
            # # print("Q:", Q_matrix)
            # # # ==== End step 4 ============================


            # ignore log(0) using 1e-8 - GPT fix
            # loss = -torch.log(ensemble_softmax.gather(1, y.unsqueeze(1)) + 1e-8).mean()
            # total_loss += loss.item() * X.size(0)

            # preds = ensemble_softmax.argmax(dim=1)

            mismatched = preds != y  # Boolean mask
            if mismatched.any():  # Check if there's at least one mismatch
                with open("/home/ma012/AlexServer/log/path_predict.txt", "a", encoding="utf-8") as file:
                    for path, mismatch in zip(paths, mismatched):
                        if mismatch:
                            file.write(f"{path}\n")


            total_correct += (preds == y).sum().item()

            total_samples += X.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write(f"\n test loss: {avg_loss} | test accuracy: {avg_acc}\n===================================================================")


    _, _, avg_sensitivity, avg_specificity = calculate_sensitivity_specificity(all_labels, all_preds, num_classes=3)


    return avg_loss, avg_acc, avg_sensitivity, avg_specificity


def max_each_ensemble(models: List[torch.nn.Module], 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module,
                       device: torch.device,log: str='log.txt') -> Tuple[float, float]:

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for model in models:
        model.eval()

    with torch.inference_mode():
        # X,y and path image
        for (X, y,paths) in dataloader:
            X, y = X.to(device), y.to(device)

            softmax_list = []
    
            for model in models:
                logits = model(X)
                softmax_out = torch.softmax(logits, dim=1) # step 1 - softmax_out_i = p_i (each classifier)

                softmax_list.append(softmax_out) 

            # ==== Weight Voting ====
            sofmax_list_tensor = torch.stack(softmax_list) #4,16,3

            max_softmax,_ = sofmax_list_tensor.max(dim=0)

            loss = loss_fn(max_softmax,y)
            total_loss += loss.item()
            
            # ====== END weight voting method ====
            
            ### == take argmax of Weight ====
            preds = max_softmax.argmax(dim=1)
            # print(f"preds:{preds}")

            mismatched = preds != y  # Boolean mask
            if mismatched.any():  # Check if there's at least one mismatch
                with open("/home/ma012/AlexServer/log/path_predict.txt", "a", encoding="utf-8") as file:
                    for path, mismatch in zip(paths, mismatched):
                        if mismatch:
                            file.write(f"{path}\n")

            total_correct += (preds == y).sum().item()

            total_samples += X.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    with open(f"{log}","a",encoding = "utf-8")  as file:
        file.write(f"\n test loss: {avg_loss} | test accuracy: {avg_acc}\n===================================================================")


    _, _, avg_sensitivity, avg_specificity = calculate_sensitivity_specificity(all_labels, all_preds, num_classes=3)


    return avg_loss, avg_acc, avg_sensitivity, avg_specificity

# ============= Inference Stage ===============

def ensemble_inference_step(models: List[torch.nn.Module], 
                         image: torch.Tensor, 
                         device: torch.device) -> Tuple[int, torch.Tensor]:
    """
    Predict the class of a single image using an ensemble of models.

    Args:
        models (List[torch.nn.Module]): List of trained models.
        image (torch.Tensor): A single image tensor of shape [1, C, H, W].
        device (torch.device): Device to run inference on.

    Returns:
        Tuple[int, torch.Tensor]: Predicted class index and softmax scores.
    """

    # Ensure all models are in evaluation mode
    for model in models:
        model.eval()

    image = image.to(device)

    with torch.inference_mode():
        ensemble_softmax = None

        for model in models:
            logits = model(image)
            softmax_out = torch.softmax(logits, dim=1)

            if ensemble_softmax is None:
                ensemble_softmax = softmax_out
            else:
                ensemble_softmax += softmax_out

        ensemble_softmax /= len(models)  # average the predictions

        predicted_class = ensemble_softmax.argmax(dim=1).item()

    return predicted_class, ensemble_softmax.squeeze()


def inference_step(model: torch.nn.Module, 
                   image: torch.Tensor, 
                   device: torch.device) -> Tuple[int, torch.Tensor]:
    """
    Predict the class of a single image using an ensemble of models.

    Args:
        model (List[torch.nn.Module]): Trained models.
        image (torch.Tensor): A single image tensor of shape [1, C, H, W].
        device (torch.device): Device to run inference on.

    Returns:
        Tuple[int, torch.Tensor]: Predicted class index and softmax scores.
    """


    image = image.to(device)

    with torch.inference_mode():

        logits = model(image)
        softmax_out = torch.softmax(logits, dim=1)

        predicted_class = softmax_out.argmax(dim=1).item()

    return predicted_class, softmax_out.squeeze()
