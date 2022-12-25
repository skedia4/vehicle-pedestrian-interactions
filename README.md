### Set Up

```bash
virtualenv -p python3 myenv
source myenv/bin/activate
pip install gym
cd ./gym-pas
pip install -e .
cd ../
pip install tensorboard
pip install matplotlib
!pip install pyyaml==5.4.1
```

Install pytorch : https://pytorch.org/get-started/locally/

In 'myenv/lib/python3.6/site-packages/gym/wrappers/order_enforcing.py', add ", collision_flag" in these two lines:
```bash
observation, reward, done, info, collision_flag = self.env.step(action)
return observation, reward, done, info, collision_flag
```

### Run the code
`source myenv/bin/activate`

train
`python src/main.py`

evaluate
`python src/main_eval.py`

### Check the training result
`tensorboard --logdir ./runs/test --port 6006`
or checkout 'checkpoint/output.log'