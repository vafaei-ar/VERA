import yaml
import datetime
import re
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Regular expressions for consent detection
AFFIRM_PATTERNS = [
    r"\b(yes|yeah|yep|ok|okay|sure|alright|agree|consent|i consent|absolutely|definitely|of course)\b",
    r"\b(i (do|will)|that's fine|sounds good|go ahead|let's do it)\b"
]

DENY_PATTERNS = [
    r"\b(no|nope|nah|decline|don't|do not|won't|will not|refuse|disagree)\b",
    r"\b(i (don't|do not|won't|will not)|not interested|not comfortable)\b"
]

AFFIRM_REGEX = re.compile("|".join(AFFIRM_PATTERNS), re.IGNORECASE)
DENY_REGEX = re.compile("|".join(DENY_PATTERNS), re.IGNORECASE)

class Dialog:
    """Dialog Finite State Machine for managing conversation flow"""
    
    def __init__(self, scenario_path: Union[str, Path], honorific: str = "Mr.", patient_name: str = "Patient"):
        self.scenario_path = Path(scenario_path)
        self.honorific = honorific
        self.patient_name = patient_name
        
        # Load scenario
        self.scenario = self._load_scenario()
        self.scenario_name = self.scenario_path.stem
        
        # Dialog state
        self.current_index = -1
        self.current_key = None
        self.current_type = None
        self.responses = {}
        self.last_prompt_text = ""
        self.wrapup_text = ""
        self.finished = False
        self.consent_given = False
        self._reprompt_text: Optional[str] = None
        self._post_answer_text: Optional[str] = None
        
        # Timeout tracking for question repetition
        self.question_start_time = None
        self.question_timeout_seconds = 20  # Increased from 10 to 20 seconds
        self.question_repeat_count = 0
        self.max_repeats = 1  # Reduced from 2 to 1 to avoid too many repetitions
        
        # Initialize wrapup text
        if "wrapup" in self.scenario:
            self.wrapup_text = self.scenario["wrapup"].get("message", "Thank you for your time.")
        
        logger.info(f"Dialog initialized with scenario: {self.scenario_name}")
    
    def start_question_timer(self):
        """Start the timer for the current question"""
        self.question_start_time = time.time()
        self.question_repeat_count = 0
    
    def check_question_timeout(self) -> bool:
        """Check if the current question has timed out and should be repeated"""
        if self.question_start_time is None:
            return False
        
        elapsed = time.time() - self.question_start_time
        return elapsed >= self.question_timeout_seconds
    
    def should_repeat_question(self) -> bool:
        """Check if we should repeat the current question"""
        if self.question_start_time is None:
            return False
        
        elapsed = time.time() - self.question_start_time
        return (elapsed >= self.question_timeout_seconds and 
                self.question_repeat_count < self.max_repeats)
    
    def repeat_question(self) -> str:
        """Repeat the current question with a timeout message"""
        if self.question_repeat_count >= self.max_repeats:
            return None
        
        self.question_repeat_count += 1
        self.question_start_time = time.time()  # Reset timer
        
        # More gentle timeout message
        timeout_message = f"I didn't hear a response. Let me repeat: {self.last_prompt_text}"
        logger.info(f"Repeating question {self.question_repeat_count}/{self.max_repeats}: {self.current_key}")
        return timeout_message
    
    def _load_scenario(self) -> Dict:
        """Load scenario from YAML file"""
        try:
            with open(self.scenario_path, 'r', encoding='utf-8') as f:
                scenario = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ["meta", "greeting", "flow"]
            for section in required_sections:
                if section not in scenario:
                    raise ValueError(f"Missing required section: {section}")
            
            logger.info(f"Loaded scenario from {self.scenario_path}")
            return scenario
            
        except Exception as e:
            logger.error(f"Failed to load scenario from {self.scenario_path}: {e}")
            raise
    
    def _get_time_of_day(self) -> str:
        """Get appropriate greeting based on time of day"""
        hour = datetime.datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    def build_greeting(self) -> str:
        """Build the greeting message with variable substitution"""
        try:
            greeting_template = self.scenario["greeting"]["template"]
            meta = self.scenario.get("meta", {})
            
            # Prepare variables for substitution
            variables = {
                "timeofday": self._get_time_of_day(),
                "honorific": self.honorific,
                "patient_name": self.patient_name,
                "organization": meta.get("organization", "Healthcare Organization"),
                "service_name": meta.get("service_name", "AI Assistant"),
                "site": meta.get("site", "Medical Center")
            }
            
            # Format the greeting
            greeting = greeting_template.format(**variables)
            self.last_prompt_text = greeting
            
            logger.info(f"Built greeting for {self.honorific} {self.patient_name}")
            return greeting
            
        except Exception as e:
            logger.error(f"Failed to build greeting: {e}")
            fallback = f"Hello {self.honorific} {self.patient_name}. How can I help you today?"
            self.last_prompt_text = fallback
            return fallback
    
    def next_prompt(self) -> Optional[str]:
        """Get the next prompt in the conversation flow"""
        if self.finished:
            return None
        
        flow = self.scenario.get("flow", [])
        
        while True:
            self.current_index += 1
            
            # Check if we've reached the end
            if self.current_index >= len(flow):
                self.finished = True
                self.current_key = None
                return None
            
            step = flow[self.current_index]
            
            # Skip section headers (they're just organizational)
            if "section" in step:
                logger.debug(f"Entering section: {step['section']}")
                continue
            
            # Process regular dialog step
            self.current_key = step.get("key")
            self.current_type = step.get("type", "free")
            prompt = step.get("prompt", "")
            
            if not self.current_key or not prompt:
                logger.warning(f"Invalid step at index {self.current_index}: missing key or prompt")
                continue
            
            self.last_prompt_text = prompt
            self.start_question_timer()  # Start timer for this question
            logger.info(f"Next prompt: {self.current_key} ({self.current_type})")
            return prompt
        
        # Shouldn't reach here, but just in case
        self.finished = True
        return None
    
    def submit_answer(self, text: str):
        """Submit an answer for the current question"""
        # If no current question yet (e.g., answer comes after greeting),
        # map this answer to the first actionable step in the flow.
        if not self.current_key:
            flow = self.scenario.get("flow", [])
            for i, step in enumerate(flow):
                if "key" in step:
                    self.current_index = i
                    self.current_key = step.get("key")
                    self.current_type = step.get("type", "free")
                    logger.info(f"Primed first step '{self.current_key}' for initial answer")
                    break
            if not self.current_key:
                logger.warning("No actionable steps found in scenario flow")
                return
        
        text = text.strip()
        logger.info(f"Answer for {self.current_key}: '{text}'")
        logger.info(f"Dialog state before answer - finished: {self.finished}, consent_given: {self.consent_given}")
        
        # Reset question timer since we received an answer
        self.question_start_time = None
        self.question_repeat_count = 0
        
        # Handle different question types
        if self.current_type == "confirm":
            self._handle_confirm_answer(text)
        else:
            # For "free" type and others, just store the answer
            self.responses[self.current_key] = text
            logger.info(f"Stored free-form answer for {self.current_key}")
        
        logger.info(f"Dialog state after answer - finished: {self.finished}, consent_given: {self.consent_given}")
    
    def _handle_confirm_answer(self, text: str):
        """Handle confirmation/yes-no type answers"""
        # Store the raw answer
        self.responses[self.current_key] = text
        
        # Check for affirmation or denial
        # Normalize text for robust matching
        norm = text.lower().strip()
        # Treat some very short affirmative tokens safely
        short_affirm = norm in {"y", "yes", "yeah", "yep", "ok", "okay", "sure", "i consent"}
        short_deny = norm in {"n", "no", "nope", "nah", "decline", "don't", "do not"}
        is_affirm = short_affirm or bool(AFFIRM_REGEX.search(norm))
        is_deny = short_deny or bool(DENY_REGEX.search(norm))
        
        if self.current_key == "consent":
            # Default: stay on consent until clear yes/no
            self._reprompt_text = None
            logger.info(f"Processing consent answer: '{text}' - is_affirm: {is_affirm}, is_deny: {is_deny}")
            
            if is_deny:
                # Consent denied: finish immediately with on_deny message if present
                self.consent_given = False
                on_deny_msg = None
                try:
                    flow = self.scenario.get("flow", [])
                    for step in flow:
                        if step.get("key") == "consent":
                            on_deny_msg = step.get("on_deny")
                            break
                except Exception:
                    on_deny_msg = None
                if on_deny_msg:
                    self.wrapup_text = on_deny_msg
                self.finished = True
                logger.info("Consent denied - ending session")
            elif is_affirm:
                # Consent affirmed: proceed; next_prompt() will advance
                self.consent_given = True
                logger.info("Consent affirmed - proceeding with assessment")
            else:
                # Unclear response: set reprompt text and keep current_key
                logger.info(f"Unclear consent response: '{text}' - re-prompting consent")
                self._reprompt_text = "Please answer yes or no. Do you consent to proceed with this recorded call?"
                # restart timer to avoid immediate timeout repeat
                self.start_question_timer()
                logger.info("Restarted question timer for consent reprompt")
        else:
            # Other confirm questions: only proceed on clear yes/no; unclear -> reprompt
            self._reprompt_text = None
            if not (is_affirm or is_deny):
                self._reprompt_text = f"Please answer yes or no. {self.last_prompt_text}"
                self.start_question_timer()
            else:
                # Queue optional follow-up info based on on_affirm/on_deny in scenario
                try:
                    flow = self.scenario.get("flow", [])
                    step = flow[self.current_index] if 0 <= self.current_index < len(flow) else {}
                    if is_deny and isinstance(step.get("on_deny"), str):
                        self._post_answer_text = step.get("on_deny")
                    elif is_affirm and isinstance(step.get("on_affirm"), str):
                        self._post_answer_text = step.get("on_affirm")
                except Exception:
                    # Ignore lookup issues
                    pass
    
    def get_current_question(self) -> Optional[Dict]:
        """Get information about the current question"""
        if not self.current_key or self.finished:
            return None
        
        return {
            "key": self.current_key,
            "type": self.current_type,
            "prompt": self.last_prompt_text,
            "index": self.current_index
        }
    
    def get_responses(self) -> Dict[str, Any]:
        """Get all collected responses"""
        return self.responses.copy()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get dialog progress information"""
        flow = self.scenario.get("flow", [])
        
        # Count actual questions (not section headers)
        total_questions = sum(1 for step in flow if "key" in step)
        answered_questions = len([k for k in self.responses.keys() if not k.endswith("_confirmed")])
        
        return {
            "current_index": self.current_index,
            "total_steps": len(flow),
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "progress_percent": (answered_questions / total_questions * 100) if total_questions > 0 else 0,
            "finished": self.finished,
            "consent_given": self.consent_given
        }

    def get_and_clear_reprompt(self) -> Optional[str]:
        """Return reprompt text if any, and clear it."""
        text = self._reprompt_text
        self._reprompt_text = None
        return text

    def get_and_clear_post_answer(self) -> Optional[str]:
        """Return optional follow-up message for the last answer, then clear it."""
        text = self._post_answer_text
        self._post_answer_text = None
        return text
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """Get information about the loaded scenario"""
        meta = self.scenario.get("meta", {})
        return {
            "name": self.scenario_name,
            "organization": meta.get("organization"),
            "service_name": meta.get("service_name"),
            "site": meta.get("site"),
            "total_steps": len(self.scenario.get("flow", [])),
            "has_greeting": "greeting" in self.scenario,
            "has_wrapup": "wrapup" in self.scenario
        }
    
    def restart(self):
        """Restart the dialog from the beginning"""
        self.current_index = -1
        self.current_key = None
        self.current_type = None
        self.responses.clear()
        self.last_prompt_text = ""
        self.finished = False
        self.consent_given = False
        
        # Reset wrapup text
        if "wrapup" in self.scenario:
            self.wrapup_text = self.scenario["wrapup"].get("message", "Thank you for your time.")
        
        logger.info("Dialog restarted")
    
    def skip_to_question(self, question_key: str) -> bool:
        """Skip to a specific question by key"""
        flow = self.scenario.get("flow", [])
        
        for i, step in enumerate(flow):
            if step.get("key") == question_key:
                self.current_index = i - 1  # Will be incremented by next_prompt()
                logger.info(f"Skipped to question: {question_key}")
                return True
        
        logger.warning(f"Question key not found: {question_key}")
        return False
    
    def export_session(self) -> Dict[str, Any]:
        """Export complete session data"""
        return {
            "scenario_name": self.scenario_name,
            "scenario_path": str(self.scenario_path),
            "patient_info": {
                "honorific": self.honorific,
                "name": self.patient_name
            },
            "dialog_state": {
                "current_index": self.current_index,
                "current_key": self.current_key,
                "finished": self.finished,
                "consent_given": self.consent_given
            },
            "responses": self.responses,
            "progress": self.get_progress(),
            "scenario_info": self.get_scenario_info()
        }

