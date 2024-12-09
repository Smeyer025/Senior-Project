import "./MenuElement.css";

export default function MenuElement(props) {
    return (
        <div className={props.className} onClick={props.onClick}>
            {props.children}
        </div>
    );
}