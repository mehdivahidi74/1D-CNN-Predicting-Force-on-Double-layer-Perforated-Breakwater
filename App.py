import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
st.markdown("""
    <h3 style='text-align: center; font-family: "Times New Roman";'>Predicting Wave Force on Double-Layer Perforated Breakwater Using 1D Convolutional Neural Networks</h3>
""", unsafe_allow_html=True)
Breakwater = Image.open('Breakwater.PNG')
st.image(Breakwater)
def user_input():
    st.markdown("""
    <h4 style='text-align: center; font-family: "Times New Roman";'>Select Variable (cm)</h4>
""", unsafe_allow_html=True)
    B = st.sidebar.slider('Chamber Width (B)',10.0 , 300.0 , 80.0, step = 1.0)
    IH = st.sidebar.slider('Impermeable Height (IH)',1.0 , 20.0 , 4.0 , step = 0.1)
    H = st.sidebar.slider('Incident Wave Height (H)',1.0 ,20.0 , 10.0 , step = 0.1)
    T = st.sidebar.slider('Wave Period (T)',0.1 , 15.0 , 2.4 , step = 0.1)
    h = st.sidebar.slider('Water depth (h)',1.0 , 100.0 , 40.0 , step = 1.0)
    return B , IH , H , T, h
B , IH , H , T, h = user_input()
T0 = []
A = np.linspace(0.005740825, 30.001225, 101)
for _ in range (len(A)):
  T0.append(A[_] /T)
st.markdown("""
    <h4 style='text-align: center; font-family: "Times New Roman";'>Non-dimensional Input Variables</h4>
""", unsafe_allow_html=True)

# Create styled HTML strings for each variable
H_h_html = f"<p style='font-size:14px; font-weight:bold; font-family:Times New Roman;'>H / h: {H/h:.2f}</p>"
B_h_html = f"<p style='font-size:14px; font-weight:bold; font-family:Times New Roman;'>B / h: {B/h:.2f}</p>"
IH_h_html = f"<p style='font-size:14px; font-weight:bold; font-family:Times New Roman;'>IH / h: {IH/h:.2f}</p>"

# Format the list T0 into a single string with styling
T0_formatted = ', '.join([f"{value:.2f}" for value in T0])
T0_html = f"<p style='font-size:14px; font-weight:bold; font-family:Times New Roman;'>T*: {T0_formatted}</p>"

# Use st.markdown to render the HTML strings
st.markdown(H_h_html, unsafe_allow_html=True)
st.markdown(B_h_html, unsafe_allow_html=True)
st.markdown(IH_h_html, unsafe_allow_html=True)
st.markdown(T0_html, unsafe_allow_html=True)
X = [H/h, B/h, IH/h,]
for i in range (len(T0)):
    X.append(T0[i])
X = np.array(X)
scaling_info_df = pd.read_excel('scaled_data_and_info.xlsx', sheet_name='Scaling Info')
mean = scaling_info_df['Mean'].values
std_dev = scaling_info_df['Standard Deviation'].values
X = (X - mean) / std_dev
X = X.reshape(1, 104, 1)
# Define custom layers and model
class Conv1DLayer:
    def __init__(self, filters, kernel_size, strides, padding, activation, weights, biases):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.weights = weights
        self.biases = biases
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        if self.padding == 'same':
            pad_total = max(self.kernel_size - self.strides, 0)
            pad_start = pad_total // 2
            pad_end = pad_total - pad_start
            x = np.pad(x, ((0, 0), (pad_start, pad_end), (0, 0)), mode='constant', constant_values=0)
        
        output = []
        for i in range(0, x.shape[1] - self.kernel_size + 1, self.strides):
            conv_region = x[:, i:i+self.kernel_size, :]
            conv_out = np.tensordot(conv_region, self.weights, axes=([1, 2], [0, 1])) + self.biases
            if self.activation == 'relu':
                conv_out = self.relu(conv_out)
            output.append(conv_out)
        
        return np.array(output).transpose(1, 0, 2)

class MaxPooling1DLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, x):
        output = []
        for i in range(0, x.shape[1] - self.pool_size + 1, self.pool_size):
            pool_region = x[:, i:i+self.pool_size, :]
            pool_out = np.max(pool_region, axis=1)
            output.append(pool_out)
        
        return np.array(output).transpose(1, 0, 2)

class FlattenLayer:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class DenseLayer:
    def __init__(self, units, activation, weights, biases):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.biases = biases
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def linear(self, x):
        return x
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.biases
        if self.activation == 'relu':
            return self.relu(output)
        elif self.activation == 'linear':
            return self.linear(output)

class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate
    
    def forward(self, x):
        return x  # No dropout during inference

class CustomModel_Front:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

# Load the Excel file into a DataFrame
df_Front = pd.read_excel('Weight_Predicting_Front.xlsx')

# Filter the DataFrame for each layer's weights and biases
conv1d_weights_Front = df_Front[(df_Front['layer'] == 'conv1d') & (df_Front['type'] == 'weight')]['weight'].values
conv1d_biases_Front = df_Front[(df_Front['layer'] == 'conv1d') & (df_Front['type'] == 'bias')]['weight'].values

dense_weights_Front = df_Front[(df_Front['layer'] == 'dense') & (df_Front['type'] == 'weight')]['weight'].values
dense_biases_Front = df_Front[(df_Front['layer'] == 'dense') & (df_Front['type'] == 'bias')]['weight'].values

dense_1_weights_Front = df_Front[(df_Front['layer'] == 'dense_1') & (df_Front['type'] == 'weight')]['weight'].values
dense_1_biases_Front = df_Front[(df_Front['layer'] == 'dense_1') & (df_Front['type'] == 'bias')]['weight'].values

# Reshape weights for Conv1D layer (filters, kernel_size, input_channels)
conv1d_weights_Front = np.reshape(conv1d_weights_Front, (23, 1, 140))
conv1d_biases_Front = np.reshape(conv1d_biases_Front, (140,))

# Reshape weights for Dense layers
dense_weights_Front = np.reshape(dense_weights_Front, (140, 70))
dense_biases_Front = np.reshape(dense_biases_Front, (70,))

dense_1_weights_Front = np.reshape(dense_1_weights_Front, (70, 101))
dense_1_biases_Front = np.reshape(dense_1_biases_Front, (101,))

# Initialize and build the custom model
custom_model_Front = CustomModel_Front()
custom_model_Front.add(Conv1DLayer(filters=140, kernel_size=23, strides=8, padding='same', activation='relu', 
                             weights=conv1d_weights_Front, biases=conv1d_biases_Front))
custom_model_Front.add(MaxPooling1DLayer(pool_size=8))
custom_model_Front.add(FlattenLayer())
custom_model_Front.add(DenseLayer(units=70, activation='relu', weights=dense_weights_Front, biases=dense_biases_Front))
custom_model_Front.add(DropoutLayer(rate=0.0))
custom_model_Front.add(DenseLayer(units=101, activation='linear', weights=dense_1_weights_Front, biases=dense_1_biases_Front))
custom_predictions_Front = custom_model_Front.forward(X)




class Conv1DLayer:
    def __init__(self, filters, kernel_size, strides, padding, activation, weights, biases):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.weights = weights
        self.biases = biases
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        if self.padding == 'same':
            pad_total = max(self.kernel_size - self.strides, 0)
            pad_start = pad_total // 2
            pad_end = pad_total - pad_start
            x = np.pad(x, ((0, 0), (pad_start, pad_end), (0, 0)), mode='constant', constant_values=0)
        
        output = []
        for i in range(0, x.shape[1] - self.kernel_size + 1, self.strides):
            conv_region = x[:, i:i+self.kernel_size, :]
            conv_out = np.tensordot(conv_region, self.weights, axes=([1, 2], [0, 1])) + self.biases
            if self.activation == 'relu':
                conv_out = self.relu(conv_out)
            output.append(conv_out)
        
        return np.array(output).transpose(1, 0, 2)

class MaxPooling1DLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, x):
        output = []
        for i in range(0, x.shape[1] - self.pool_size + 1, self.pool_size):
            pool_region = x[:, i:i+self.pool_size, :]
            pool_out = np.max(pool_region, axis=1)
            output.append(pool_out)
        
        return np.array(output).transpose(1, 0, 2)

class FlattenLayer:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class DenseLayer:
    def __init__(self, units, activation, weights, biases):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.biases = biases
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def linear(self, x):
        return x
    
    def forward(self, x):
        output = np.dot(x, self.weights) + self.biases
        if self.activation == 'relu':
            return self.relu(output)
        elif self.activation == 'linear':
            return self.linear(output)

class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate
    
    def forward(self, x):
        return x  # No dropout during inference

class CustomModel_Back:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

# Load the Excel file into a DataFrame
df_Back = pd.read_excel('Weight_Predicting_Back.xlsx')

# Filter the DataFrame for each layer's weights and biases
conv1d_weights_Back = df_Back[(df_Back['layer'] == 'conv1d') & (df_Back['type'] == 'weight')]['weight'].values
conv1d_biases_Back = df_Back[(df_Back['layer'] == 'conv1d') & (df_Back['type'] == 'bias')]['weight'].values

dense_weights_Back = df_Back[(df_Back['layer'] == 'dense') & (df_Back['type'] == 'weight')]['weight'].values
dense_biases_Back = df_Back[(df_Back['layer'] == 'dense') & (df_Back['type'] == 'bias')]['weight'].values

dense_1_weights_Back = df_Back[(df_Back['layer'] == 'dense_1') & (df_Back['type'] == 'weight')]['weight'].values
dense_1_biases_Back = df_Back[(df_Back['layer'] == 'dense_1') & (df_Back['type'] == 'bias')]['weight'].values

# Reshape weights for Conv1D layer (filters, kernel_size, input_channels)
conv1d_weights_Back = np.reshape(conv1d_weights_Back, (23, 1, 140))
conv1d_biases_Back = np.reshape(conv1d_biases_Back, (140,))

# Reshape weights for Dense layers
dense_weights_Back = np.reshape(dense_weights_Back, (140, 70))
dense_biases_Back = np.reshape(dense_biases_Back, (70,))

dense_1_weights_Back = np.reshape(dense_1_weights_Back, (70, 101))
dense_1_biases_Back = np.reshape(dense_1_biases_Back, (101,))

# Initialize and build the custom model
custom_model_Back = CustomModel_Back()
custom_model_Back.add(Conv1DLayer(filters=140, kernel_size=23, strides=8, padding='same', activation='relu', 
                             weights=conv1d_weights_Back, biases=conv1d_biases_Back))
custom_model_Back.add(MaxPooling1DLayer(pool_size=8))
custom_model_Back.add(FlattenLayer())
custom_model_Back.add(DenseLayer(units=70, activation='relu', weights=dense_weights_Back, biases=dense_biases_Back))
custom_model_Back.add(DropoutLayer(rate=0.0))
custom_model_Back.add(DenseLayer(units=101, activation='linear', weights=dense_1_weights_Back, biases=dense_1_biases_Back))

# Make predictions using the custom model
custom_predictions_back = custom_model_Back.forward(X)
import matplotlib.pyplot as plt
# Function to create the plot

def create_plot(custom_predictions_Front, custom_predictions_back, T0):
    # Find max values and their indices
    max_force_front = np.max(custom_predictions_Front)
    max_force_back = np.max(custom_predictions_back)
    max_index_front = np.argmax(custom_predictions_Front)
    max_index_back = np.argmax(custom_predictions_back)
    x_axis = np.array(T0)
    # Get corresponding x-axis values
    x_value_front = x_axis[max_index_front]
    x_value_back = x_axis[max_index_back]

    # Calculate the absolute difference between x-axis values
    x_difference = abs(x_value_back - x_value_front)

    # Plotting
    plt.figure(figsize=(30, 15))
    plt.plot(x_axis, custom_predictions_Front.flatten(), label='Force on Front Perforated Wall', color='blue', marker='o')
    plt.plot(x_axis, custom_predictions_back.flatten(), label='Force on Back Perforated Wall', color='red', marker='s')

    # Highlighting the maximum points
    plt.scatter(x_value_front, max_force_front, color='yellow', s=1000, facecolors='yellow', edgecolors='black', linewidth=2, label='Max Force Front')
    plt.scatter(x_value_back, max_force_back, color='limegreen', s=1000, facecolors='limegreen', edgecolors='black', linewidth=2, label='Max Force Back')

    # Plot a horizontal line at the average y position of the max forces
    y_avg = np.maximum(max_force_front, max_force_back)
    plt.plot([x_value_front, x_value_back], [y_avg, y_avg], color='black', linestyle='--', linewidth=5)

    # Annotate the difference on the plot
    plt.annotate(f'Δt = {x_difference:.2f}', 
                 xy=((x_value_front + x_value_back) / 2, y_avg),
                 xytext=(0, 10), textcoords='offset points', 
                 ha='center', fontsize=30, fontweight='bold', fontname='Times New Roman')

    # Set labels and title
    plt.xlabel('Non-dimensional Time Series', fontsize=30, fontweight='bold', fontname='Times New Roman')
    plt.ylabel(r'Force (N/m$^{2}$)', fontsize=30, fontweight='bold', fontname='Times New Roman')
    plt.title('Force on Perforated Walls Over Time', fontsize=30, fontweight='bold', fontname='Times New Roman')
    
    # Customize legend
    plt.legend(prop={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}, loc='upper left', frameon=True, borderpad=1, handletextpad=1, labelspacing=2.0)

    # Customize x and y axis tick labels
    plt.xticks(fontsize=20, fontweight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=20, fontweight='bold', fontname='Times New Roman')

    plt.grid(True)
    
    return plt

# Create the plot
plt = create_plot(custom_predictions_Front, custom_predictions_back, T0)

# Display the plot in Streamlit
st.pyplot(plt)

# Get the maximum wave force and format it to 2 decimal places
max_wave_force_Front = np.max(custom_predictions_Front)
formatted_wave_force_Front = f"{max_wave_force_Front:.2f}"

# Use HTML to set the font to Times New Roman and include N/m² with the power of 2, also specify font size
st.markdown(
    f"<div style='text-align: center; font-family: Times New Roman; font-size: 24px;font-weight: bold;'>"
    f"Maximum Wave Force on Front Perforated wall: {formatted_wave_force_Front} N/m<sup>2</sup>"
    "</div>",
    unsafe_allow_html=True)

max_wave_force_Back = np.max(custom_predictions_back)
formatted_wave_force_Back = f"{max_wave_force_Back:.2f}"

# Use HTML to set the font to Times New Roman and include N/m² with the power of 2, also specify font size
st.markdown(
    f"<div style='text-align: center; font-family: Times New Roman; font-size: 24px;font-weight: bold;'>"
    f"Maximum Wave Force on Back Perforated wall: {formatted_wave_force_Back} N/m<sup>2</sup>"
    "</div>",
    unsafe_allow_html=True)
