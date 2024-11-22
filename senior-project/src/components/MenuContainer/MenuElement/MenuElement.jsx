import "./MenuElement.css";

export default function MenuElement(props) {
    return (
        <div className={props.className}>
            {props.children}
        </div>
    );
}