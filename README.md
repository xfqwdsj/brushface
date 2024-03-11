## BrushFace

[![Stars](https://img.shields.io/github/stars/xfqwdsj/brushface?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/xfqwdsj/brushface/stargazers)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](LICENSE)

**⚠️ This project is under development and the API may change at any time. Besides, the repository may be force pushed
at any time. Please do not use it in production environments.**

BrushFace is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for
python. It is a hybrid face recognition framework wrapping some **state-of-the-art** models, allowing users to extend
the models easily or use their own models.

Experiments show that human beings have 97.53% accuracy on facial recognition tasks whereas those models already reached
and passed that accuracy level.

## Why the name so weird?

The name "BrushFace" is a combination of "Brush" and "Face". "Brush" is directly translated from "刷" in Chinese, which
means "scan" in combination with "face" (or "脸"). "刷脸" is a popular term in China that means "face recognition". The
name "BrushFace" is a tribute to Chinese culture and Chinese developers.

Chinese also combine "刷" with other words to express the meaning of "do something quickly". For example, "刷卡" means
"swipe a card".

Of course, "刷" is also used with its original meaning of "brush" in Chinese. For example, "刷牙" means "brush teeth".

## Credits

This project is inspired by the following projects:

- [serengil/deepface](https://github.com/serengil/deepface) (fork source)

## Licence

BrushFace is licensed under the MIT License - see [`LICENSE`](LICENSE) for more details.

BrushFace wraps some external face recognition models. Besides, age, gender and race / ethnicity models were trained on
the backbone of VGG-Face with transfer learning. Licence types will be inherited if you are going to use those models.
Please check the license types of those models for production purposes.
