# mjpc_robot_traning_model

## Steps to Build and Run the Model

### Step 1: Clone the Repo
```
git clone https://github.com/Vinaysangwan/mjpc_robot_traning_model
```

### Step2: Create Build Directory
```
mkdir build && cd build
```

### Step3: Build CMake
```
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMJPC_BUILD_GRPC_SERVICE=ON
```

### Step4: Compile & Run
```
cmake --build . -j8
./bin/mjpc --task="Grab"
```

## Note: 
Run the Step4 each time you update the code. Rest of the steps are to be performed only once.
