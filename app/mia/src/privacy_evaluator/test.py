
def train_defend_ppb(self, train_loader, log_pref="", defend_arg=0.01):
    self.model.train()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        self.optimizer.zero_grad()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss1 = self.criterion(outputs, targets)
        ranked_outputs, _ = torch.topk(outputs, self.num_cls, dim=-1)
        size = targets.size(0)
        even_size = size // 2 * 2
        if even_size > 0:
            loss2 = F.kl_div(F.log_softmax(ranked_outputs[:even_size // 2], dim=-1),
                                F.softmax(ranked_outputs[even_size // 2:even_size], dim=-1),
                                reduction='batchmean')
        else:
            loss2 = torch.zeros(1).to(self.device)
        loss = loss1 + defend_arg * loss2
        total_loss += loss.item() * size
        total_loss1 += loss1.item() * size
        total_loss2 += loss2.item() * size
        total += size
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        loss.backward()
        self.optimizer.step()
    acc = 100. * correct / total
    total_loss /= total
    total_loss1 /= total
    total_loss2 /= total

    if self.scheduler:
        self.scheduler.step()
    if log_pref:
        print("{}: Accuracy {:.3f}, Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}".format(
            log_pref, acc, total_loss, total_loss1, total_loss2))
    return acc, total_loss

def train_defend_adv(self, train_loader, test_loader, log_pref="", privacy_theta=0.5):
    """
    modified from
    https://github.com/Lab41/cyphercat/blob/master/Defenses/Adversarial_Regularization.ipynb
    """
    total_loss = 0
    correct = 0
    total = 0
    infer_iterations = 7

    # train adversarial network
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    train_iter2 = iter(train_loader)

    self.model.eval()
    self.attack_model.train()
    for infer_iter in range(infer_iterations):
        with torch.no_grad():
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            in_predicts = F.softmax(self.model(inputs), dim=-1)
            in_targets = F.one_hot(targets, num_classes=self.num_cls).float()

            try:
                inputs, targets = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                inputs, targets = next(test_iter)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out_predicts = F.softmax(self.model(inputs), dim=-1)
            out_targets = F.one_hot(targets, num_classes=self.num_cls).float()

            infer_train_data = torch.cat([torch.cat([in_predicts, in_targets], dim=-1),
                                            torch.cat([out_predicts, out_targets], dim=-1)], dim=0)
            infer_train_label = torch.cat([torch.ones(in_predicts.size(0)),
                                            torch.zeros(out_predicts.size(0))]).long().to(self.device)

        self.attack_model_optim.zero_grad()
        infer_loss = privacy_theta * F.cross_entropy(self.attack_model(infer_train_data), infer_train_label)
        infer_loss.backward()
        self.attack_model_optim.step()

    self.model.train()
    self.attack_model.eval()
    try:
        inputs, targets = next(train_iter2)
    except StopIteration:
        train_iter2 = iter(train_loader)
        inputs, targets = next(train_iter2)
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.model(inputs)
    loss1 = self.criterion(outputs, targets)
    in_predicts = F.softmax(outputs, dim=-1)
    in_targets = F.one_hot(targets, num_classes=self.num_cls).float()
    infer_data = torch.cat([in_predicts, in_targets], dim=-1)
    infer_labels = torch.ones(targets.size(0)).long().to(self.device)
    infer_loss = F.cross_entropy(self.attack_model(infer_data), infer_labels)
    loss = loss1 - privacy_theta * infer_loss
    loss.backward()
    self.optimizer.step()
    total_loss += loss.item() * targets.size(0)
    total += targets.size(0)
    _, predicted = outputs.max(1)
    correct += predicted.eq(targets).sum().item()
    if self.scheduler:
        self.scheduler.step()
    acc = 100. * correct / total
    total_loss /= total
    if log_pref:
        print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
    return acc, total_loss


def run():
    if args.defense == 'adv':
        target_model_save_folder = save_dir + "/target_models_adv"
        if not os.path.exists(target_model_save_folder):
            os.makedirs(target_model_save_folder)
        model = ResNet18(device, save_dir, num_cls, epochs, lr=0.01, weight_decay=5e-4, input_dim=100, dropout = 0)
        best_acc = 0
        count = 0
        for epoch in range(epochs):
            train_acc, train_loss = model.train_defend_adv(target_train_loader, target_test_loader, f"epoch {epoch} train")
            val_acc, val_loss = model.test(target_valid_loader, f"epoch {epoch} valid")
            test_acc, test_loss = model.test(target_test_loader, f"epoch {epoch} test")
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = model.save(epoch, test_acc, test_loss)
                best_path = save_path
                count = 0
            elif early_stop > 0:
                count += 1
                if count > early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break
        shutil.copyfile(best_path, f"{target_model_save_folder}/best.pth")


    elif args.defense == 'ppb':
        target_model_save_folder = save_dir + "/target_models_adv"
        if not os.path.exists(target_model_save_folder):
            os.makedirs(target_model_save_folder)
        model = ResNet18(device, save_dir, num_cls, epochs, lr=0.01, weight_decay=5e-4, input_dim=100, dropout = 0)
        best_acc = 0
        count = 0
        for epoch in range(epochs):
            train_acc, train_loss = model.train_defend_ppb(target_train_loader, target_test_loader, f"epoch {epoch} train")
            val_acc, val_loss = model.test(target_valid_loader, f"epoch {epoch} valid")
            test_acc, test_loss = model.test(target_test_loader, f"epoch {epoch} test")
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = model.save(epoch, test_acc, test_loss)
                best_path = save_path
                count = 0
            elif early_stop > 0:
                count += 1
                if count > early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break
        shutil.copyfile(best_path, f"{target_model_save_folder}/best.pth")