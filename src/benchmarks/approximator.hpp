#ifndef __APPROXIMATOR_H__
#define __APPROXIMATOR_H__

namespace approximator {

  template <typename T>
    inline bool isApprox(T number) {
      return (number & (1 << (sizeof(T) * 8 - 1)));
    }

  template <typename T>
    inline T value(T number) {
      return (number & (~(1 << (sizeof(T) * 8 - 1))));
    }

  template <typename T>
    inline void setValue(T& number, T value) {
      T tmp = (number & (1 << (sizeof(T) * 8 - 1)));
      std::cout << "[[ tmp = " << tmp << " ]] ";
      number = (value | (number & (1 << (sizeof(T) * 8 - 1))));
    }

  template <typename T>
    inline void setApprox(T& number) {
      number |= (1 << (sizeof(T) * 8 - 1));
    }

  template <typename T>
    inline void unsetApprox(T& number) {
      number &= (~(1 << (sizeof(T) * 8 - 1)));
    }

}

#endif // __APPROXIMATOR_HPP__
