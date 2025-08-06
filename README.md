# Simple Language v2.0

A beginner-friendly programming language that transpiles to C++. Simple combines the ease of Python/Lua with the performance of C++, making it perfect for kids learning programming and professionals who want to save time.

## 🚀 Features

- **Minimal syntax**: No braces required, uses indentation and `end` keywords
- **Arrays & Vectors**: Built-in support for dynamic arrays with C++ vector performance
- **String interpolation**: Python-style `{variable}` syntax in strings
- **Enhanced control flow**: Full if/elif/else chains, multiple loop types
- **Auto typing**: Variables are automatically typed unless explicitly specified
- **Easy output**: Simple `say` keyword with interpolation support
- **C++ compatibility**: Transpiles to clean, optimized C++ code
- **Commercial ready**: Global variables, function parameters, recursion

## 🎯 Design Philosophy

**Easy for kids** 👶 - Simple syntax, no complex symbols, intuitive flow
**Powerful for pros** 💪 - Full C++ performance, time-saving

## 📚 Syntax Overview

### Includes
```simple
use "iostream"
use "vector"
use "string"
```

### Functions with Parameters
```simple
fn main
    say "Hello, world!"
end

fn add a b
    return a + b
end

fn fibonacci n
    if n <= 1
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end
```

### Variables & Arrays
```simple
# Simple variables
x = 5
name = "Alice"

# Arrays (become C++ vectors)
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]

# Array access
first = numbers[0]
say "First number:", first
```

### String Interpolation
```simple
name = "Developer"
version = 2.0
say "Welcome to Simple v{version}, {name}!"
```

### Enhanced Control Flow
```simple
# Full if-elif-else chains
score = 85
if score >= 90
    say "Grade: A"
elif score >= 80
    say "Grade: B"
elif score >= 70
    say "Grade: C"
else
    say "Grade: F"
end

# Multiple loop types
# Traditional counting loop
loop i 1 10
    say "Count:", i
end

# For-in loop (range-based)
for item in numbers
    say "Processing:", item
end

# While loop
counter = 0
while counter < 5
    say "Counter:", counter
    counter = counter + 1
end
```

### Global Variables
```simple
# Global configuration
app_name = "My App"
version = 1.0

fn main
    say "Running {app_name} v{version}"
end
```

## 💡 Example Programs

### Basic Example
**hello.simple:**
```simple
use "iostream"

fn main
    name = "World"
    say "Hello, {name}!"
    
    # Array operations
    numbers = [1, 2, 3, 4, 5]
    for num in numbers
        say "Number: {num}"
    end
    
    # Function call
    result = add(10, 20)
    say "10 + 20 = {result}"
end

fn add a b
    return a + b
end
```

### Commercial Example
**data_processor.simple:**
```simple
use "iostream"
use "vector"

# Global configuration
app_name = "Data Processor Pro"
version = 2.1

fn process_scores scores
    total = 0
    max_score = 0
    
    for score in scores
        total = total + score
        if score > max_score
            max_score = score
        end
    end
    
    average = total / scores.size()
    say "Total: {total}, Average: {average}, Max: {max_score}"
    return average
end

fn grade_analysis scores
    for score in scores
        if score >= 90
            say "Score {score}: Excellent!"
        elif score >= 80
            say "Score {score}: Good"
        elif score >= 70
            say "Score {score}: Average"
        else
            say "Score {score}: Needs improvement"
        end
    end
end

fn main
    say "=== {app_name} v{version} ==="
    
    # Process student scores
    test_scores = [95, 87, 92, 78, 88, 96, 82]
    
    say "Processing {test_scores.size()} scores..."
    avg = process_scores(test_scores)
    
    say "Grade Analysis:"
    grade_analysis(test_scores)
    
    say "Analysis complete!"
end
```

**Generated C++ (excerpt):**
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

auto app_name = "Data Processor Pro";
auto version = 2.1;

auto process_scores(auto scores) {
    auto total = 0;
    auto max_score = 0;
    for (auto score : scores) {
        auto total = total + score;
        if (score > max_score) {
            auto max_score = score;
        }
    }
    auto average = total / scores.size();
    cout << "Total: " << total << ", Average: " << average << ", Max: " << max_score << endl;
    return average;
}

int main() {
    cout << "=== " << app_name << " v" << version << " ===" << endl;
    vector<int> test_scores = {95, 87, 92, 78, 88, 96, 82};
    // ... rest of the code
    return 0;
}
```

## 🛠️ Usage

### Quick Start
```bash
# Transpile and compile in one step
python compile_simple_v2.py my_program.simple

# Run the generated executable
my_program.exe
```

### Advanced Usage
```bash
# Transpile only (generate C++ without compiling)
python compile_simple_v2.py program.simple --transpile-only

# Custom executable name
python compile_simple_v2.py program.simple my_app.exe

# Direct transpilation
python simple_transpiler_v2.py input.simple output.cpp
```

### Development Workflow
1. Write your `.simple` file
2. Run `python compile_simple_v2.py yourfile.simple`
3. Execute the generated `.exe`
4. Iterate and improve!

## 📋 Requirements

- **Python 3.6+** - For running the transpiler
- **Visual Studio Build Tools** - For C++ compilation
- **Developer Command Prompt** - Run compilation from VS Developer Command Prompt

### Installation
1. Install Python 3.6+
2. Install Visual Studio Build Tools or Visual Studio Community
3. Open "Developer Command Prompt for VS"
4. Navigate to your Simple Language directory
5. Start coding!

## 📖 Language Reference

### Core Rules
1. **Indentation**: Use consistent indentation (spaces or tabs)
2. **Blocks**: End blocks with the `end` keyword
3. **Functions**: Start with `fn` keyword, followed by name and parameters
4. **Variables**: Declared by assignment, automatically typed
5. **Arrays**: Use `[item1, item2, item3]` syntax
6. **Strings**: Support interpolation with `{variable}` syntax
7. **Comments**: Start with `#`

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `and`, `or`, `not` (planned)

### Built-in Methods
- **Arrays**: `.size()`, `.push_back(item)`, `[index]` access
- **Strings**: `.length()`, `.upper()`, `.lower()` (planned)

## 🔧 Conversion to C++

| Simple | C++ |
|--------|-----|
| `use "lib"` | `#include <lib>` + `using namespace std;` |
| `fn name params` | `auto name(auto params)` with forward declarations |
| `say "text"` | `cout << "text" << endl;` |
| `say "Hello {name}"` | `cout << "Hello " << name << endl;` |
| `[1, 2, 3]` | `vector<int> {1, 2, 3}` |
| `loop i 1 10` | `for (auto i = 1; i <= 10; i++)` |
| `for item in array` | `for (auto item : array)` |
| `while condition` | `while (condition)` |
| `if/elif/else` | `if/else if/else` with proper braces |

## 📁 Project Structure

```
simple-language/
├── simple_transpiler_v2.py      # Enhanced transpiler engine
├── compile_simple_v2.py         # CLI compiler with features
├── simple_transpiler.py         # Original basic transpiler
├── compile_simple.py            # Original basic compiler
├── commercial_demo.simple       # Full-featured demo
├── basic_enhanced_test.simple   # Enhanced features test
├── test.simple                  # Basic functionality test
└── README.md                    # This documentation
```

## 🎯 Use Cases

### For Kids & Beginners
- **Learn programming concepts** without complex syntax
- **Immediate feedback** with simple compilation
- **Visual results** with easy output statements
- **Gradual complexity** from simple scripts to full programs

### For Professionals
- **Rapid prototyping** of algorithms
- **Quick scripting** for data processing
- **Teaching tool** for explaining concepts
- **C++ performance** without the boilerplate
- **Time-saving** for simple utilities

## 🚀 Future Roadmap

### v2.1 (Planned)
- ✅ Classes and objects
- ✅ Method definitions
- ✅ Constructor support
- ✅ Object-oriented programming

### v2.2 (Planned)
- 📋 Error handling (try/catch)
- 📋 Standard library functions
- 📋 File I/O operations
- 📋 More string methods

### v3.0 (Vision)
- 📋 IDE integration
- 📋 Debugging support
- 📋 Package management
- 📋 Cross-platform compilation
- 📋 Web assembly target

## 🤝 Contributing

Simple Language is designed to be educational and practical. Contributions welcome for:
- New language features
- Better error messages
- Performance optimizations
- Documentation improvements
- Example programs

---

**Simple Language**: Making programming accessible to everyone, from kids taking their first steps to professionals who value their time. 🌟
