# 💡 Project Ideas - Build Something Awesome

**⏱️ 10 min read | Pick your project**

**This document provides 10 detailed project ideas** with complete specifications, starter templates, and difficulty ratings. For additional inspiration and API examples, see the comprehensive research document.

---

## 📖 Before You Start: Know Your Tools

**Read the comprehensive research first:**  
👉 [**OpenAI Tools Research (October 2025)**](openai_tools_research_oct2025.md)

This document covers all available OpenAI APIs including:
- **Latest Tools:** GPT-5 Pro, Sora 2, AgentKit, o4-mini, gpt-realtime mini
- **Core APIs:** Chat Completion, Vision, DALL-E 3, Whisper, TTS, Embeddings
- **Specialized:** Assistants API, Function Calling, Moderation API
- **Additional Project Inspiration** by API type
- **Code Examples** and best practices for each API

**Quick API Reference:**
- **Text/Chat** → Chat Completion (GPT-4o, GPT-5 Pro)
- **Images (Analyze)** → Vision API
- **Images (Generate)** → DALL-E 3
- **Audio (Listen)** → Whisper
- **Audio (Speak)** → TTS
- **Search/Match** → Embeddings
- **Agents/Tools** → Assistants API
- **Safety** → Moderation API
- **Video (NEW!)** → Sora 2

---

## 🎯 How to Choose

### Ask Yourself:

1. **What interests me?** (Pick something you'd actually use)
2. **What OpenAI tool fits?** (Match tool to problem)
3. **Can I finish in 2 weeks?** (Be realistic)
4. **Will it showcase my skills?** (Think demo day)

### Red Flags:

- ❌ "Build ChatGPT" (too vague, too big)
- ❌ "AI that does everything" (scope too large)
- ❌ "Requires a database" (too complex for 2 weeks)
- ❌ "Need to learn React first" (too many new things)

### Green Flags:

- ✅ Clear, specific problem to solve
- ✅ Uses 1-2 OpenAI APIs
- ✅ Can demo in 5 minutes
- ✅ Can build with Python CLI

---

## 📊 Project Difficulty Levels

### 🟢 Starter (Great for beginners)
- **Time:** 1-1.5 weeks
- **Tests:** 15-25 tests
- **APIs:** 1 main OpenAI API
- **Good if:** First time with TDD or AI APIs

### 🟡 Intermediate (Recommended)
- **Time:** 1.5-2 weeks
- **Tests:** 25-40 tests
- **APIs:** 1-2 OpenAI APIs
- **Good if:** Comfortable with Python, want to be challenged

### 🔴 Advanced (For the bold)
- **Time:** Full 2 weeks
- **Tests:** 40+ tests
- **APIs:** 2-3 OpenAI APIs
- **Good if:** Want to push yourself, experienced coder

---

## 🟢 Starter Projects

### 1. Code Explainer 📚

**What:** CLI that explains code snippets

**How it works:**
```bash
$ python -m src.main explain "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"

Explanation:
This is a recursive function that calculates factorial...
Time Complexity: O(n)
Space Complexity: O(n) due to recursion stack...
```

**OpenAI APIs:** Chat Completion

**Components:**
- CodeExplainer model (stores code + explanation)
- ExplainerClient (calls OpenAI)
- ExplainerParser (formats response)
- CLI interface

**Why good:**
- Single API call per request
- Clear input/output
- Easy to test with mocks
- Practical for learning

---

### 2. Language Translator with Detection 🌍

**What:** Translate text and detect source language

**How it works:**
```bash
$ python -m src.main translate "Bonjour le monde"

Detected: French
English: Hello world
Spanish: Hola mundo
German: Hallo Welt
```

**OpenAI APIs:** Chat Completion

**Components:**
- Translation model (source, target, text)
- TranslationClient (calls OpenAI)
- TranslationParser (extracts translations)
- CLI with multiple target languages

**Why good:**
- Single API, creative prompting
- Clear success criteria
- Fun to demo
- Practical application

---

### 3. Sentiment Analyzer 😊😐😢

**What:** Analyze sentiment of text (social media, reviews)

**How it works:**
```bash
$ python -m src.main analyze "This product is amazing! Best purchase ever!"

Sentiment: Positive (95% confidence)
Emotion: Joy, Excitement
Key phrases: "amazing", "best purchase"
Recommendation: Feature in marketing
```

**OpenAI APIs:** Chat Completion

**Components:**
- SentimentResult model
- SentimentClient (calls OpenAI)
- SentimentParser (extracts sentiment data)
- CLI with batch processing

**Why good:**
- Well-defined output format
- Easy to validate
- Lots of test cases
- Business-relevant

---

## 🟡 Intermediate Projects (Recommended)

### 4. Smart Recipe Generator 👨‍🍳

**What:** Generate recipes from ingredients you have

**How it works:**
```bash
$ python -m src.main recipe "chicken, rice, tomatoes"

Recipe: Mediterranean Chicken Bowl
Ingredients needed:
- Chicken (you have)
- Rice (you have)
- Tomatoes (you have)
- Olive oil
- Lemon

Instructions:
1. Season chicken with...
2. Cook rice with...

Nutrition: [calories, protein, etc.]
Shopping list: [missing ingredients]
```

**OpenAI APIs:** Chat Completion + (optional) DALL-E for image

**Components:**
- Ingredient model
- Recipe model
- RecipeClient (calls OpenAI)
- RecipeParser (structured recipe data)
- NutritionCalculator
- ShoppingListGenerator

**Why good:**
- Multiple components to test
- Clear value proposition
- Fun to demo
- Room for creativity

---

### 5. Image Analysis Reporter 🖼️

**What:** Analyze images and generate detailed reports

**How it works:**
```bash
$ python -m src.main analyze image.jpg

Analysis Report
==============
Type: Product Photo
Objects detected: Laptop, Coffee Mug, Notebook
Scene: Office/Workspace
Mood: Professional, Organized
Colors: Navy blue (dominant), White, Gray

Suggestions:
- Good lighting
- Consider adding plants for warmth
- Background slightly cluttered

SEO Keywords: laptop, workspace, home office, productivity
```

**OpenAI APIs:** Vision (image analysis) + Chat (report generation)

**Components:**
- ImageAnalysis model
- VisionClient (analyzes image)
- ReportGenerator (formats findings)
- ImageParser (extracts structured data)
- CLI with file upload

**Why good:**
- Two APIs working together
- Visual demo (show images)
- Impressive results
- Professional application

---

### 6. Meeting Summarizer 📝

**What:** Transcribe audio and generate meeting summaries

**How it works:**
```bash
$ python -m src.main summarize meeting.mp3

Transcript:
[Full transcript...]

Summary:
Key Points:
- Project deadline moved to next Friday
- Sarah will handle design
- Budget approved for $5000

Action Items:
- John: Submit proposal by Wed
- Sarah: Share mockups by Thurs
- Team: Review budget spreadsheet

Decisions Made:
- Approved vendor A over vendor B
- Will use TypeScript for new features

Next Meeting: Thursday 2pm
```

**OpenAI APIs:** Whisper (transcription) + Chat (summarization)

**Components:**
- Audio model
- Transcript model
- Summary model
- WhisperClient (transcription)
- SummarizerClient (generates summary)
- AudioParser (handles audio files)

**Why good:**
- Two distinct APIs
- Clear pipeline (audio → text → summary)
- Professional use case
- Impressive demo

---

### 7. Code Review Assistant 🔍

**What:** Automated code review with suggestions

**How it works:**
```bash
$ python -m src.main review src/mycode.py

Code Review Report
==================
Overall Quality: B+ (Good)

Issues Found:
🔴 CRITICAL (1)
- Line 23: SQL injection vulnerability
  Fix: Use parameterized queries

🟡 WARNING (3)
- Line 45: Function too complex (15 branches)
  Suggestion: Split into smaller functions
- Line 67: Unused variable 'temp'
- Line 89: Missing error handling

✅ GOOD (5)
- Well-documented functions
- Good test coverage
- Clear naming conventions

Suggestions:
- Add type hints
- Extract validation logic
- Consider caching results
```

**OpenAI APIs:** Chat Completion

**Components:**
- CodeFile model
- ReviewIssue model
- CodeReviewer client
- ReviewParser (structures findings)
- SeverityClassifier
- RecommendationGenerator

**Why good:**
- Complex parsing (severity levels, line numbers)
- Multiple output formats
- Very practical
- Great for demo

---

## 🔴 Advanced Projects

### 8. Content Repurposer 📱➡️📝➡️🎬

**What:** Convert content between formats (blog → tweets → video script)

**How it works:**
```bash
$ python -m src.main repurpose blog-post.md

Generated Content
================

📱 Twitter Thread (5 tweets):
1/5 [First tweet...]
2/5 [Second tweet...]
...

📝 LinkedIn Post:
[Professional version...]

🎬 YouTube Script (8 min):
[INTRO]
[MAIN POINTS]
[CALL TO ACTION]

📧 Email Newsletter:
Subject: [catchy subject]
Body: [formatted for email]
```

**OpenAI APIs:** Chat Completion (multiple specialized prompts)

**Components:**
- Content model (source content)
- TwitterThread model
- LinkedInPost model
- VideoScript model
- ContentRepurposer (orchestrates conversions)
- PlatformAdapter (adapts for each platform)
- FormatValidator (ensures platform limits)

**Why good:**
- Multiple transformations
- Complex state management
- Real business value
- Very demo-able

---

### 9. AI Tutor with Personalization 👨‍🏫

**What:** Personalized tutoring that adapts to student

**How it works:**
```bash
$ python -m src.main tutor

AI Tutor: What would you like to learn about?
You: Python functions

AI: Great! Let me check your level first.
[Asks diagnostic questions]

AI: I see you're intermediate. Let's dive into decorators.
[Teaches concept]

AI: Let's practice. Try writing a decorator that times functions.
You: [writes code]

AI: Good start! But there's an issue on line 3...
[Explains with example]

Progress:
Functions: ████████░░ 80%
Classes: ████████░░ 75%
Decorators: ███░░░░░░░ 30% (current)
```

**OpenAI APIs:** Chat Completion (conversational), TTS (optional voice)

**Components:**
- Student model (tracks progress)
- LearningPath model
- Conversation model
- TutorClient (manages chat)
- ProgressTracker (tracks learning)
- DifficultyCalibratorr
- QuizGenerator
- ExplanationAdapter

**Why good:**
- Stateful conversation
- Multiple features
- Complex testing (conversation flow)
- Impressive capabilities

---

### 10. Multi-Modal Story Generator 📖🖼️

**What:** Generate illustrated stories with narration

**How it works:**
```bash
$ python -m src.main story --theme "space adventure" --protagonist "robot"

Generating Story...

Title: "Zara's Cosmic Journey"

[Chapter 1: The Discovery]
Zara, a curious maintenance robot, discovers an ancient map...

[Generated image: robot-looking-at-stars.png]

[Audio: chapter1.mp3 - narrated version]

[Chapter 2: The Launch]
Against all protocols, Zara sneaks aboard the departing ship...

[Generated image: robot-in-spaceship.png]

[Audio: chapter2.mp3]

Story Statistics:
- Word count: 2,450
- Reading time: 10 minutes
- Images: 5
- Audio length: 12:30
```

**OpenAI APIs:** Chat (story), DALL-E (images), TTS (narration)

**Components:**
- Story model (chapters, characters)
- Chapter model
- Character model
- StoryGenerator (creates narrative)
- IllustrationGenerator (DALL-E)
- NarrationGenerator (TTS)
- StoryOrchestrator (coordinates all)
- ThemeValidator
- ContentPipeline

**Why good:**
- Uses 3 OpenAI APIs
- Complex coordination
- Beautiful output
- Memorable demo

---

## 🎨 Custom Project? Here's the Framework

### Step 1: Define the Problem

**Template:**
```
I want to build a [type] that [does what] for [who/what].

Example: I want to build a CLI tool that generates workout plans 
for people with limited equipment.
```

---

### Step 2: Choose OpenAI API(s)

**Match problem to API:**
- Text generation → Chat Completion
- Images → DALL-E or Vision
- Audio → Whisper or TTS
- Code → Chat Completion (optimized prompt)

---

### Step 3: Break Down Components

**Template:**
```
Data Models:
- Model 1 (what data?)
- Model 2 (what data?)

API Clients:
- Client 1 (calls which API?)
- Client 2 (calls which API?)

Business Logic:
- Service 1 (does what?)
- Parser 1 (parses what?)

CLI:
- What commands?
- What arguments?
```

---

### Step 4: Validate Scope

**Checklist:**
- [ ] Can I explain it in 1 sentence?
- [ ] Can I build 3+ models?
- [ ] Can I write 20+ tests?
- [ ] Can I demo in 5 minutes?
- [ ] Can I finish in 2 weeks?

**If any NO → simplify!**

---

## 🎯 Project Success Checklist

Before committing to a project:

### Technical:
- [ ] Clear API endpoints needed
- [ ] Can mock API responses for tests
- [ ] Can break into testable components
- [ ] No external dependencies (databases, etc.)

### Practical:
- [ ] Can run from command line
- [ ] Clear input/output
- [ ] Easy to demo (5 min max)
- [ ] Fits 2-week timeline

### Learning:
- [ ] Uses TDD effectively
- [ ] Showcases testing skills
- [ ] Demonstrates API integration
- [ ] Shows error handling

### Grading:
- [ ] 25+ tests possible
- [ ] Multiple components
- [ ] Good documentation potential
- [ ] Clear git commits (20+)

---

## 💡 Pro Tips for Project Selection

### Tip #1: Pick Something You'd Use
**Why:** You'll be more motivated, better at testing edge cases

**Example:** Love cooking? → Recipe generator
**Example:** Struggle with writing? → Content assistant

---

### Tip #2: Start with Starter, Upgrade if Time
**Why:** Better to finish a simple project than abandon complex one

**Plan:**
- Week 1: Build starter version
- Week 2: If ahead, add features

---

### Tip #3: Think About the Demo
**Why:** Grading includes demo quality

**Good demo:**
- Shows real input/output
- Explains one interesting technical choice
- Shows test coverage
- Runs smoothly

**Bad demo:**
- Too complex to explain
- Breaks during demo
- All code, no results

---

### Tip #4: Ask "What Could Go Wrong?"
**Why:** Good projects handle errors gracefully

**Questions:**
- What if API is down?
- What if input is empty?
- What if response is malformed?
- What if rate limit hit?

**These become your test cases!**

---

## 🆘 Project Questions to Ask Claude

### Starting Out:
```
"I want to build [project idea].

Help me:
1. List the components I'll need
2. Suggest a good project structure
3. Identify what to build first
4. List potential test cases

Use the enterprise_ai_demo1_websearch repo as a reference."
```

### During Development:
```
"I'm building [project].
So far I have: [list components]

I'm stuck on [specific issue].
Here's my code: [paste relevant code]

How should I approach this?"
```

### Before Demo:
```
"Review my project for demo day.

Components: [list]
Tests: [count]
Coverage: [percentage]

Suggestions:
1. What's most impressive to highlight?
2. What should I mention about technical challenges?
3. How can I make demo smooth?"
```

---

## 📊 Project Complexity Comparison

| Project | Difficulty | APIs | Models | Est. Tests | Wow Factor |
|---------|-----------|------|--------|-----------|-----------|
| Code Explainer | 🟢 Starter | 1 | 2-3 | 15-20 | ⭐⭐ |
| Translator | 🟢 Starter | 1 | 3-4 | 18-25 | ⭐⭐ |
| Sentiment Analysis | 🟢 Starter | 1 | 3-4 | 20-25 | ⭐⭐⭐ |
| Recipe Generator | 🟡 Intermediate | 1-2 | 4-5 | 25-35 | ⭐⭐⭐⭐ |
| Image Reporter | 🟡 Intermediate | 2 | 4-6 | 30-40 | ⭐⭐⭐⭐ |
| Meeting Summarizer | 🟡 Intermediate | 2 | 5-6 | 30-40 | ⭐⭐⭐⭐⭐ |
| Code Reviewer | 🟡 Intermediate | 1 | 5-7 | 35-45 | ⭐⭐⭐⭐ |
| Content Repurposer | 🔴 Advanced | 1 | 6-8 | 40-50 | ⭐⭐⭐⭐⭐ |
| AI Tutor | 🔴 Advanced | 1-2 | 7-9 | 45-55 | ⭐⭐⭐⭐⭐ |
| Story Generator | 🔴 Advanced | 3 | 8-10 | 50-60 | ⭐⭐⭐⭐⭐ |

---

## 🎯 Grading Alignment

### How Projects Map to Grading (100 points):

**Functionality (25 points):**
- Works as specified ✅
- Handles errors gracefully ✅
- Good user experience ✅

**Testing (25 points):**
- 90%+ coverage ✅
- Good test names ✅
- Tests edge cases ✅

**Code Quality (20 points):**
- TDD throughout ✅
- Clean structure ✅
- Follows patterns ✅
- Good documentation ✅

**Git (10 points):**
- 20+ commits ✅
- Conventional commits ✅
- Atomic commits ✅

**Documentation (10 points):**
- Clear README ✅
- Setup instructions ✅
- Usage examples ✅

**Demo (10 points):**
- Smooth presentation ✅
- Explains tech challenges ✅
- Shows working code ✅

**All project ideas above support getting full points!**

---

## 🚀 Ready to Start?

### Next Steps:

1. **Read the comprehensive research** - [OpenAI Tools Research](openai_tools_research_oct2025.md) (20 min)
2. **Read through project ideas here** (10 min)
3. **Pick your top 3 ideas** (5 min)
4. **Deep dive into your chosen API** - Use research doc for details (15 min)
5. **Sketch components on paper** (10 min)
6. **Discuss with instructor** (Session 1)
7. **Start with tests!** (Session 2)

---

## 📚 Resources

### 🌟 Start Here:
**[OpenAI Tools Research (October 2025)](openai_tools_research_oct2025.md)** - Complete reference with:
- Latest API updates (GPT-5 Pro, Sora 2, AgentKit)
- Detailed capabilities for each API
- 60+ project ideas organized by API
- Code examples and best practices
- Pricing and rate limit information

### OpenAI Official Documentation:
- Chat Completion: https://platform.openai.com/docs/guides/text-generation
- Vision: https://platform.openai.com/docs/guides/vision
- DALL-E: https://platform.openai.com/docs/guides/images
- Whisper: https://platform.openai.com/docs/guides/speech-to-text
- TTS: https://platform.openai.com/docs/guides/text-to-speech

### Example Repository:
- This repo: enterprise_ai_demo1_websearch
- Study: src/client.py, src/parser.py, tests/

### Get Help:
- Claude: "Help me design [project]"
- Instructor: In-class support
- Classmates: Pair programming

---

**Pick a project you're excited about. You'll be working on it for 2 weeks!** 🚀

---

*For TDD workflow, see [TDD_WORKFLOW.md](TDD_WORKFLOW.md)*
*For git workflow, see [GIT_WORKFLOW.md](GIT_WORKFLOW.md)*
*For AI collaboration, see [AI_COLLABORATION.md](AI_COLLABORATION.md)*
